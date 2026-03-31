import json
import re
from pathlib import Path

from discord.ext import commands, tasks
from loguru import logger

from sigrok import genai
from sigrok.config import settings
from sigrok.genai import SIGROK_PERSONALITY_SYSTEM_PROMPT
from sigrok.social_client import BlueskyClient, BlueskyNotification


class BlueskyCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.client = BlueskyClient(settings.social.bluesky)
        self._state_path = Path(settings.social.bluesky.state_file)
        self._processed_uris: list[str] = []
        self._bootstrapped = False
        self._own_handle = settings.social.bluesky.identifier.strip().lower()
        self._load_state()
        if settings.social.bluesky.enabled:
            self.poll_mentions.start()
        else:
            logger.info("Bluesky integration disabled; mention poller not started.")

    def cog_unload(self) -> None:
        if self.poll_mentions.is_running():
            self.poll_mentions.cancel()

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to read Bluesky state file {self._state_path}: {exc}")
            return

        self._bootstrapped = bool(data.get("bootstrapped"))
        raw_processed = data.get("processed_uris") or []
        self._processed_uris = [
            str(uri) for uri in raw_processed if isinstance(uri, str)
        ][-500:]

    def _save_state(self) -> None:
        payload = {
            "bootstrapped": self._bootstrapped,
            "processed_uris": self._processed_uris[-500:],
        }
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _is_processed(self, uri: str) -> bool:
        return uri in self._processed_uris

    def _mark_processed(self, uri: str) -> None:
        if self._is_processed(uri):
            return
        self._processed_uris.append(uri)
        if len(self._processed_uris) > 500:
            self._processed_uris = self._processed_uris[-500:]

    def _strip_bot_mention(self, text: str) -> str:
        handle = self._own_handle.lstrip("@").strip()
        if not handle:
            return " ".join(text.split()).strip()
        pattern = rf"(?<!\w)@{re.escape(handle)}\b"
        stripped = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        return " ".join(stripped.split()).strip(" ,:\n\t")

    def _normalize_bot_response(self, text: str) -> str:
        normalized = text.strip()
        if (
            len(normalized) >= 2
            and normalized[0] == normalized[-1]
            and normalized[0] in {'"', "'"}
        ):
            normalized = normalized[1:-1].strip()
        normalized = re.sub(r"\s+\n", "\n", normalized)
        normalized = re.sub(r"^sigrok:\s*", "", normalized, flags=re.IGNORECASE).strip()
        normalized = self._strip_bot_mention(normalized)
        return normalized

    def _should_skip_response(self, question: str, response: str) -> bool:
        r = response.strip()
        return (
            not r
            or r.lower() == question.lower().strip()
            or (len(r) < 200 and r in SIGROK_PERSONALITY_SYSTEM_PROMPT)
            or r in {"not worth my time", "I couldn't answer that right now."}
        )

    async def _ensure_handle(self) -> str:
        if self._own_handle:
            return self._own_handle
        self._own_handle = (await self.client.get_own_handle()).strip().lower()
        return self._own_handle

    async def _bootstrap_notifications(self) -> None:
        notifications = await self.client.list_notifications(limit=50)
        for notification in notifications:
            if notification.reason == "mention":
                self._mark_processed(notification.uri)
        self._bootstrapped = True
        self._save_state()
        try:
            await self.client.mark_notifications_seen()
        except Exception as exc:
            logger.warning(f"Failed to mark Bluesky notifications seen during bootstrap: {exc}")
        logger.info(
            "Bootstrapped Bluesky mention poller with {} historical mention(s).",
            len([n for n in notifications if n.reason == "mention"]),
        )

    async def _reply_to_notification(
        self, notification: BlueskyNotification, own_handle: str
    ) -> bool:
        thread = await self.client.get_post_thread(notification.uri)
        if not thread:
            logger.warning(f"Skipping Bluesky notification {notification.uri}: empty thread")
            return True

        current_post = thread[-1]
        root_post = thread[0]
        if current_post.author_handle.lower() == own_handle.lower():
            logger.info(f"Skipping self-authored Bluesky mention {current_post.uri}")
            return True

        question = self._strip_bot_mention(current_post.text)
        if not question:
            response = "say something after the @"
        else:
            response = await genai.client.answer_social_question(
                platform="bluesky",
                account_handle=own_handle,
                question=question,
                messages=[post.to_genai_message() for post in thread[:-1]],
                current_message=current_post.to_genai_message(),
                max_chars=settings.social.bluesky.max_chars,
            )
            response = self._normalize_bot_response(response)
            if self._should_skip_response(question, response):
                logger.info(
                    "Skipping Bluesky reply for {} due to empty/invalid model output.",
                    current_post.uri,
                )
                return True

        reply_uri = await self.client.reply_to_post(
            response,
            parent=current_post,
            root=root_post,
        )
        logger.info(
            "Replied to Bluesky mention {} from @{} with {}",
            current_post.uri,
            current_post.author_handle,
            reply_uri,
        )
        return True

    @tasks.loop(seconds=settings.social.bluesky.poll_seconds)
    async def poll_mentions(self) -> None:
        if not settings.social.bluesky.enabled:
            return
        try:
            own_handle = await self._ensure_handle()
            if not self._bootstrapped:
                await self._bootstrap_notifications()
                return

            notifications = await self.client.list_notifications(limit=50)
            pending = [
                notification
                for notification in notifications
                if notification.reason == "mention" and not self._is_processed(notification.uri)
            ]
            pending.sort(key=lambda notification: notification.indexed_at)

            processed_any = False
            for notification in pending:
                handled = await self._reply_to_notification(notification, own_handle)
                if handled:
                    self._mark_processed(notification.uri)
                    processed_any = True

            if processed_any:
                self._save_state()
                try:
                    await self.client.mark_notifications_seen()
                except Exception as exc:
                    logger.warning(f"Failed to mark Bluesky notifications seen: {exc}")
        except Exception as exc:
            logger.exception(f"Unhandled Bluesky mention polling error: {exc}")

    @poll_mentions.before_loop
    async def before_poll_mentions(self) -> None:
        await self.bot.wait_until_ready()


def setup(bot: commands.Bot) -> None:
    bot.add_cog(BlueskyCog(bot))
