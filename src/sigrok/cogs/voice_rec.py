"""Voice channel capture in time-bounded chunks (WAV per speaker) for downstream transcription."""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import discord
from discord import Message, sinks
from discord.ext import commands
from loguru import logger

from sigrok.config import settings

if TYPE_CHECKING:
    from discord.ext.commands import Bot


def _guild_whitelisted(guild_id: int) -> bool:
    return any(entry.guild == guild_id for entry in settings.bot.whitelist)


def _recordings_dir() -> Path:
    base = Path(settings.bot.temp_dir)
    sub = settings.bot.voice_record.directory.strip("/\\") or "voice_recordings"
    path = base / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def _announcement_path() -> Path | None:
    raw = settings.bot.voice_record.announcement_file.strip()
    if not raw:
        return None
    return Path(raw).expanduser()


@dataclass
class _GuildSession:
    guild_id: int
    vc: discord.VoiceClient
    chunk_seconds: float
    out_dir: Path
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    chunk_index: int = 0
    driver_task: asyncio.Task[None] | None = None
    announcement_task: asyncio.Task[None] | None = None


class VoiceRecCog(commands.Cog):
    """Join voice and rotate sink-based recordings on a fixed interval."""

    _JOIN_RE = re.compile(r"\bjoin\s+us\b", re.IGNORECASE)
    _LEAVE_RE = re.compile(r"\bleave\s+us\b", re.IGNORECASE)

    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self._sessions: dict[int, _GuildSession] = {}

    def cog_unload(self) -> None:
        for sess in list(self._sessions.values()):
            sess.stop_event.set()

    async def _finalize_chunk(self, session: _GuildSession, sink: sinks.Sink) -> None:
        ts = int(time.time())
        ext = getattr(sink, "encoding", None) or "wav"
        saved: list[Path] = []
        for user_id, audio in sink.audio_data.items():
            try:
                audio.file.seek(0)
                raw = audio.file.read()
            except OSError as exc:
                logger.warning("Could not read audio for user {}: {}", user_id, exc)
                continue
            if not raw:
                continue
            path = (
                session.out_dir
                / f"g{session.guild_id}_c{session.chunk_index}_t{ts}_u{user_id}.{ext}"
            )
            path.write_bytes(raw)
            saved.append(path)
        session.chunk_index += 1
        if saved:
            logger.info(
                "Voice chunk {} guild {} — saved {} file(s): {}",
                session.chunk_index - 1,
                session.guild_id,
                len(saved),
                ", ".join(str(p) for p in saved),
            )
        await self.on_voice_chunk_saved(saved)

    async def on_voice_chunk_saved(self, paths: list[Path]) -> None:
        """Override or monkeypatch for transcription pipelines; default is no-op."""
        return None

    async def _play_announcement_once(
        self, session: _GuildSession, path: Path
    ) -> None:
        if not session.vc.is_connected():
            return
        if session.vc.is_playing():
            logger.info(
                "Skipping recording announcement in guild {} because audio is already playing.",
                session.guild_id,
            )
            return

        play_done = asyncio.Event()

        def after(exc: Exception | None) -> None:
            if exc is not None:
                logger.error(
                    "Recording announcement playback failed in guild {}: {}",
                    session.guild_id,
                    exc,
                )
            self.bot.loop.call_soon_threadsafe(play_done.set)

        try:
            source = discord.FFmpegPCMAudio(str(path))
            session.vc.play(source, after=after)
            logger.info(
                "Played recording announcement in guild {} from {}",
                session.guild_id,
                path,
            )
        except Exception as exc:
            logger.error(
                "Could not start recording announcement in guild {}: {}",
                session.guild_id,
                exc,
            )
            return

        await play_done.wait()

    async def _announcement_driver(self, session: _GuildSession, path: Path) -> None:
        interval = max(
            5.0,
            float(settings.bot.voice_record.announcement_interval_seconds),
        )
        try:
            while not session.stop_event.is_set():
                try:
                    await asyncio.wait_for(session.stop_event.wait(), timeout=interval)
                    break
                except asyncio.TimeoutError:
                    await self._play_announcement_once(session, path)
        except Exception:
            logger.exception(
                "Recording announcement driver failed for guild {}",
                session.guild_id,
            )

    async def _chunk_driver(self, session: _GuildSession) -> None:
        vc = session.vc
        try:
            while not session.stop_event.is_set():
                chunk_done = asyncio.Event()

                async def on_chunk_done(
                    s: sinks.Sink,
                    sess: _GuildSession,
                    ev: asyncio.Event,
                ) -> None:
                    try:
                        await self._finalize_chunk(sess, s)
                    finally:
                        ev.set()

                sink = sinks.WaveSink()
                vc.start_recording(
                    sink,
                    on_chunk_done,
                    session,
                    chunk_done,
                )
                try:
                    await asyncio.wait_for(
                        session.stop_event.wait(),
                        timeout=session.chunk_seconds,
                    )
                    vc.stop_recording()
                    await chunk_done.wait()
                    break
                except asyncio.TimeoutError:
                    vc.stop_recording()
                    await chunk_done.wait()
        except Exception:
            logger.exception("Voice chunk driver failed for guild {}", session.guild_id)
        finally:
            if session.announcement_task is not None:
                session.announcement_task.cancel()
                try:
                    await session.announcement_task
                except asyncio.CancelledError:
                    pass
            if vc.is_connected():
                await vc.disconnect()
            self._sessions.pop(session.guild_id, None)

    async def _start_recording_session(
        self, guild: discord.Guild, member: discord.Member
    ) -> str | None:
        """Begin chunk recording from the member's current voice channel. None = success."""
        if not discord.voice_client.has_nacl:
            return (
                "Voice requires PyNaCl. Install the `voice` extra for py-cord (see pyproject)."
            )
        if guild.id in self._sessions:
            return "Already recording in this server."
        voice = member.voice
        if not voice or not voice.channel:
            return "Join a voice channel first — I connect to the channel you're in."
        channel = voice.channel
        if not isinstance(channel, discord.VoiceChannel):
            return "That isn't a normal voice channel I can join."

        me = guild.me
        if me is None:
            return "Could not resolve bot member."
        perms = channel.permissions_for(me)
        if not perms.connect or not perms.speak:
            return "I need **Connect** and **Speak** in that voice channel."

        try:
            vc = await channel.connect()
        except Exception as exc:
            logger.error("Voice connect failed: {}", exc)
            return f"Could not connect to voice: {exc}"

        chunk = float(settings.bot.voice_record.chunk_seconds)
        if chunk < 5:
            chunk = 5.0
        session = _GuildSession(
            guild_id=guild.id,
            vc=vc,
            chunk_seconds=chunk,
            out_dir=_recordings_dir(),
        )
        announcement = _announcement_path()
        if announcement is not None:
            if not announcement.exists():
                await vc.disconnect()
                return f"Announcement file does not exist: `{announcement}`"
            if shutil.which("ffmpeg") is None:
                await vc.disconnect()
                return "ffmpeg is required to play the recording announcement audio."
            session.announcement_task = asyncio.create_task(
                self._announcement_driver(session, announcement)
            )
        session.driver_task = asyncio.create_task(self._chunk_driver(session))
        self._sessions[guild.id] = session
        return None

    async def _stop_recording_session(self, guild: discord.Guild) -> str | None:
        """Stop recording and leave voice. None = success."""
        session = self._sessions.get(guild.id)
        if not session:
            return "I'm not recording voice in this server."
        session.stop_event.set()
        if session.driver_task:
            try:
                await asyncio.wait_for(session.driver_task, timeout=120.0)
            except asyncio.TimeoutError:
                logger.error(
                    "Voice driver did not finish within timeout for guild {}", guild.id
                )
                return "I tried to stop, but the voice task did not shut down cleanly."
        return None

    def _mention_voice_action(self, message: Message) -> str | None:
        """Return 'join' or 'leave' when a mention asks for voice control."""
        text = (message.clean_content or "").lower()
        has_join = bool(self._JOIN_RE.search(text))
        has_leave = bool(self._LEAVE_RE.search(text))
        if has_join and has_leave:
            return "both"
        if has_join:
            return "join"
        if has_leave:
            return "leave"
        return None

    async def handle_mention_voice_phrase(self, message: Message) -> bool:
        """If message is a @bot join/leave voice phrase, act and return True (misc should skip genai)."""
        if message.guild is None or self.bot.user is None:
            return False
        if self.bot.user not in message.mentions:
            return False
        if not _guild_whitelisted(message.guild.id):
            return False

        action = self._mention_voice_action(message)
        if action is None:
            return False

        if action == "both":
            await message.reply(
                "Pick one: ask me to **join us** or **leave us**.", mention_author=False
            )
            return True

        if not isinstance(message.author, discord.Member):
            return False

        if action == "leave":
            err = await self._stop_recording_session(message.guild)
            if err:
                await message.reply(err, mention_author=False)
            else:
                await message.reply("Left voice — recording stopped.", mention_author=False)
            return True

        err = await self._start_recording_session(message.guild, message.author)
        if err:
            await message.reply(err, mention_author=False)
        else:
            chunk = int(max(5, float(settings.bot.voice_record.chunk_seconds)))
            vch = message.author.voice.channel if message.author.voice else None
            ch_name = vch.name if vch else "voice"
            announcement = _announcement_path()
            notice = ""
            if announcement is not None:
                interval = int(
                    max(5, settings.bot.voice_record.announcement_interval_seconds)
                )
                notice = f" I'll play the recording notice every **{interval}**s."
            await message.reply(
                f"In **{ch_name}** — recording chunks every **{chunk}**s. "
                f"Ping me with **leave us** when you're done.{notice}",
                mention_author=False,
            )
        return True


def setup(bot: Bot) -> None:
    bot.add_cog(VoiceRecCog(bot))
