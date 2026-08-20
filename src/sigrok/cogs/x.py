import asyncio
import json
import re
from pathlib import Path

from discord.ext import commands, tasks
from loguru import logger

from sigrok import genai
from sigrok.config import settings
from sigrok.genai import SIGROK_PERSONALITY_SYSTEM_PROMPT
from sigrok.social_client import XClient, XTweet
from sigrok.streaming.response import apply_nsfw_filter


class XCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.client = XClient(settings.social.x)
        self._state_path = Path(settings.social.x.state_file)
        self._processed_ids: list[str] = []
        self._since_id = ""
        self._bootstrapped = False
        self._own_username = settings.social.x.username.strip().lstrip("@").lower()
        self._load_state()
        if settings.social.x.enabled:
            self.poll_mentions.start()
        else:
            logger.info("X integration disabled; mention poller not started.")

    def cog_unload(self) -> None:
        if self.poll_mentions.is_running():
            self.poll_mentions.cancel()
        if self.client._http is not None and not self.client._http.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.client.close())
                else:
                    loop.run_until_complete(self.client.close())
            except Exception as exc:
                logger.warning(f"Failed to close X HTTP session: {exc}")

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to read X state file {self._state_path}: {exc}")
            return

        self._bootstrapped = bool(data.get("bootstrapped"))
        self._since_id = str(data.get("since_id") or "")
        raw_processed = data.get("processed_ids") or []
        self._processed_ids = [str(tweet_id) for tweet_id in raw_processed if tweet_id][-500:]

    def _save_state(self) -> None:
        payload = {
            "bootstrapped": self._bootstrapped,
            "since_id": self._since_id,
            "processed_ids": self._processed_ids[-500:],
        }
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _is_processed(self, tweet_id: str) -> bool:
        return tweet_id in self._processed_ids

    def _mark_processed(self, tweet_id: str) -> None:
        if self._is_processed(tweet_id):
            return
        self._processed_ids.append(tweet_id)
        if len(self._processed_ids) > 500:
            self._processed_ids = self._processed_ids[-500:]
        if not self._since_id or int(tweet_id) > int(self._since_id):
            self._since_id = tweet_id

    def _strip_bot_mention(self, text: str) -> str:
        handle = self._own_username.lstrip("@").strip()
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

    async def _ensure_username(self) -> str:
        if self._own_username:
            return self._own_username
        self._own_username = (await self.client.get_own_username()).strip().lower()
        return self._own_username

    async def _bootstrap_mentions(self) -> None:
        mentions = await self.client.list_mentions(max_results=50)
        for mention in mentions:
            self._mark_processed(mention.id)
        self._bootstrapped = True
        self._save_state()
        logger.info(
            "Bootstrapped X mention poller with {} historical mention(s).",
            len(mentions),
        )

    async def _reply_to_mention(self, mention: XTweet, own_username: str) -> bool:
        thread = await self.client.get_reply_chain(mention)
        if not thread:
            logger.warning(f"Skipping X mention {mention.id}: empty thread")
            return True

        current_tweet = thread[-1]
        if current_tweet.author_username.lower() == own_username.lower():
            logger.info(f"Skipping self-authored X mention {current_tweet.id}")
            return True

        question = self._strip_bot_mention(current_tweet.text)
        if not question:
            response = "say something after the @"
        else:
            response = await genai.client.answer_social_question(
                platform="x",
                account_handle=own_username,
                question=question,
                messages=[tweet.to_genai_message() for tweet in thread[:-1]],
                current_message=current_tweet.to_genai_message(),
                max_chars=settings.social.x.max_chars,
            )
            response = apply_nsfw_filter(self._normalize_bot_response(response))
            if self._should_skip_response(question, response):
                logger.info(
                    "Skipping X reply for {} due to empty/invalid model output.",
                    current_tweet.id,
                )
                return True

        reply_url = await self.client.reply_to_tweet(
            response,
            in_reply_to_tweet_id=current_tweet.id,
        )
        logger.info(
            "Replied to X mention {} from @{} with {}",
            current_tweet.id,
            current_tweet.author_username,
            reply_url,
        )
        return True

    @tasks.loop(seconds=settings.social.x.poll_seconds)
    async def poll_mentions(self) -> None:
        if not settings.social.x.enabled:
            return
        try:
            own_username = await self._ensure_username()
            if not self._bootstrapped:
                await self._bootstrap_mentions()
                return

            mentions = await self.client.list_mentions(
                since_id=self._since_id or None,
                max_results=50,
            )
            pending = [
                mention
                for mention in mentions
                if not self._is_processed(mention.id)
            ]
            pending.sort(key=lambda mention: int(mention.id))

            processed_any = False
            for mention in pending:
                handled = await self._reply_to_mention(mention, own_username)
                if handled:
                    self._mark_processed(mention.id)
                    processed_any = True

            if processed_any:
                self._save_state()
        except Exception as exc:
            logger.exception(f"Unhandled X mention polling error: {exc}")

    @poll_mentions.before_loop
    async def before_poll_mentions(self) -> None:
        await self.bot.wait_until_ready()


def setup(bot: commands.Bot) -> None:
    bot.add_cog(XCog(bot))
