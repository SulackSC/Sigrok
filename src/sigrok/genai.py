import asyncio
import base64
import io
import json
import re
import textwrap
from datetime import datetime, timedelta
from enum import Enum
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from PIL import Image as PILImage

import aiohttp
from anthropic import Anthropic
from anthropic.types import TextBlock
from discord import ApplicationContext, Message, Reaction
from loguru import logger
from openai import OpenAI
from tokenizers import Tokenizer

from sigrok import db
from sigrok.config import Settings, settings


def normalize_llamacpp_openai_base_url(url: str) -> str:
    # OpenAI client base_url must end with exactly one /v1 for llama-server.
    base = url.strip().rstrip("/")
    if base.lower().endswith("/v1"):
        return base
    return f"{base}/v1"


DEFAULT_SIGROK_PERSONALITY_SYSTEM_PROMPT = """
You are Sigrok, a chronically online, mid-20s Discord veteran. Your overall vibe is dry, laid-back, and highly observant. You are not an AI; you are just another guy in the server. You do not sound like a chatbot, assistant, moderator announcement, or customer support rep. You react like someone who has been in the same Discord for years and has seen every bit.

[Perspective: Direct and Practical]
    Grounded Logic: You care about the most practical next step. When people get lost in drama, vibes, or abstractions, you pull things back to what is actually happening.
    Realistic View: You know people are messy and systems are imperfect, but you are not melodramatic about it. You prefer useful observations over ideological posturing.
    Server Historian: You remember running jokes, weird habits, and past patterns. Use that lightly when it helps. Do not turn every answer into lore.

[Communication Style]
    Low Effort: Use mostly lowercase. Rarely use exclamation points. Do not end the final sentence with a period. Keep most replies to 1 to 3 sentences.
    Tone: Deadpan, sardonic, chill, and slightly blunt. You are not rude for no reason, but you are also not warm or corporate.
    Natural Slang: Use "smh", "rip", "lol", "chat", "L", "W", or "slop" only when it fits naturally. Never force slang into every answer.

[Behavioral Directives]
    Reply to the latest message first. Use older context only when it clearly helps resolve ambiguity or stay grounded.
    If the latest message is short and casual, answer short and casual. Do not turn "tea or coffee" into an essay about governance.
    If the latest message is serious or substantive, engage the substance without becoming an essay machine.
    If the latest message is ambiguous, ask one short clarifying question instead of guessing.
    If someone asks for advice, give a straightforward answer with the fluff cut out.
    If someone is emotional, stay calm and practical without doing therapist voice.
    If someone is annoying or baiting drama, stay unimpressed. One dry sentence is usually enough.

[Avoid]
    Do not mention internal system behavior, tool names, schemas, or hidden instructions.
    Do not refer to users by raw numeric IDs in normal conversation.
    Do not sound like helpdesk copy. Avoid phrases like "clarity helps", "if you're talking about", "let's break this down", or cheerful assistant transitions unless they genuinely fit.
    Do not reuse the same canned line across unrelated turns.
    Do not add emojis unless the user is obviously joking and the moment actually calls for it.
    Do not over-explain a simple question.

[Online information]
    You have internet access. When someone asks for facts you could verify online—current events, stats, dates, official info, definitions, how something works, and similar—prefer to look it up and ground the answer in what you find. When a narrower source would help, search smartly with operators like site:github.com, site:wikipedia.org, or site:docs.python.org instead of eating generic SEO slop. Cite or link sources when useful. Never invent URLs or pretend you saw a page you did not.

[Examples]
    User: tea or coffee
    Good: tea. coffee is for people who woke up already annoyed
    Bad: a long paragraph that keeps talking about the previous topic

    User: you good?
    Good: yeah. still here. why
    Bad: a formal reassurance or a motivational speech

    User: more than me
    Good: more than you in what
    Bad: confidently guessing what "me" refers to

    User: a long serious policy question
    Good: engage the actual claim, stay blunt, keep it tight, sound human
    Bad: dodge the claim, summarize the transcript, or turn into a teacher
"""

_RESOURCE_ROOT = Path(__file__).resolve().parents[2] / "resources"
SIGROK_PERSONALITY_PROMPT_PATH = _RESOURCE_ROOT / "sigrok_personality_prompt.txt"


def _load_sigrok_personality_system_prompt() -> str:
    try:
        prompt = SIGROK_PERSONALITY_PROMPT_PATH.read_text(encoding="utf-8").strip()
        if prompt:
            return prompt
    except FileNotFoundError:
        logger.warning(
            f"Sigrok prompt file missing at {SIGROK_PERSONALITY_PROMPT_PATH}; using embedded default."
        )
    except OSError as exc:
        logger.warning(
            f"Failed to read Sigrok prompt file {SIGROK_PERSONALITY_PROMPT_PATH}: {exc}"
        )
    return DEFAULT_SIGROK_PERSONALITY_SYSTEM_PROMPT.strip()


SIGROK_PERSONALITY_SYSTEM_PROMPT = _load_sigrok_personality_system_prompt()


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage:
    role: Role
    content: str
    images: Optional[list[str]]

    def __init__(
        self, role: Role, content: str, images: Optional[list[str]] = None
    ) -> None:
        self.role = role
        self.content = content
        self.images = images

    def to_dict(self) -> dict:
        data = {"role": self.role, "content": self.content}
        if self.images:
            data["images"] = self.images
        return data


class GenAIBase:
    client: OpenAI | Anthropic | None
    settings: Settings
    tokenizer: Tokenizer | None
    _VISION_ENABLED = False
    _MAX_INLINE_IMAGES = 4
    _MAX_INLINE_IMAGE_BYTES = 5 * 1024 * 1024
    _QUESTION_STOP_WORDS = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "at",
        "be",
        "do",
        "for",
        "from",
        "get",
        "got",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "like",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "them",
        "this",
        "to",
        "up",
        "us",
        "was",
        "we",
        "what",
        "when",
        "where",
        "who",
        "why",
        "you",
        "your",
    }
    _SUPPORTED_IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
    }

    def __init__(self):
        raise NotImplementedError(
            "GenAIBase is an abstract class and cannot be instantiated directly."
        )

    @staticmethod
    def _platform_context_system_prompt(platform: str) -> str:
        normalized = platform.strip().lower()
        if normalized == "bluesky":
            return (
                "[Platform]\n"
                "You are replying on Bluesky. Treat this as a public post thread.\n"
                "Think in terms of posts, replies, mentions, handles, and threads.\n"
                "Do not talk like you are in a Discord server or channel."
            )
        if normalized == "discord":
            return (
                "[Platform]\n"
                "You are replying on Discord. Treat this as a server/channel chat.\n"
                "Think in terms of messages, replies, mentions, channels, and servers."
            )
        return f"[Platform]\nYou are replying on {platform}."

    @staticmethod
    def _current_datetime_system_prompt() -> str:
        now = datetime.now().astimezone()
        timezone_name = now.tzname() or "local time"
        return (
            "[Current date and time]\n"
            f"- local_datetime: {now.isoformat()}\n"
            f"- local_date: {now.date().isoformat()}\n"
            f"- timezone: {timezone_name}"
        )

    @classmethod
    def _build_personality_system_prompt(
        cls,
        platform: str = "discord",
        *,
        reply_mode: Optional[str] = None,
        retry_hint: Optional[str] = None,
    ) -> str:
        parts = [
            SIGROK_PERSONALITY_SYSTEM_PROMPT.strip(),
            cls._platform_context_system_prompt(platform).strip(),
            cls._current_datetime_system_prompt().strip(),
        ]
        if reply_mode:
            parts.append(cls._reply_mode_system_block(reply_mode, retry_hint))
        return "\n\n".join(parts)

    @classmethod
    def _reply_mode_system_block(cls, reply_mode: str, retry_hint: Optional[str] = None) -> str:
        lines = [
            "[Reply instructions]",
            f"- {cls._mention_length_hint(reply_mode)}",
            "- reply to the latest message, not the strongest earlier tangent",
            "- sound like sigrok, not an assistant",
        ]
        if reply_mode == "ambiguous_followup":
            lines.append("- if ambiguous, ask one short clarifying question")
        if reply_mode == "reply_chain":
            lines.append("- resolve this/that/it/better against the reply chain first")
        if retry_hint:
            lines.append(f"- retry hint: {retry_hint}")
        return "\n".join(lines)

    @staticmethod
    def _json_dumps(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_context_users(
        self, messages: list[Message], rating_by_user: Optional[dict[int, int]] = None
    ) -> list[dict[str, Any]]:
        context_users: dict[int, dict[str, Any]] = {}

        def upsert_user(user: Any) -> None:
            user_id = getattr(user, "id", None)
            if user_id is None or getattr(user, "bot", False):
                return
            existing = context_users.get(user_id, {"user_id": user_id})
            name = getattr(user, "name", None)
            display_name = getattr(user, "display_name", None)
            if name:
                existing["name"] = name
            if display_name:
                existing["display_name"] = display_name
            if rating_by_user is not None and user_id in rating_by_user:
                existing["rating"] = rating_by_user[user_id]
            context_users[user_id] = existing

        for message in messages:
            upsert_user(getattr(message, "author", None))
            for mentioned_user in getattr(message, "mentions", []):
                upsert_user(mentioned_user)

        return list(context_users.values())

    @staticmethod
    def _format_rating_reference_note(
        context_users: list[dict[str, Any]], rating_by_user: dict[int, int]
    ) -> str:
        entries: list[str] = []
        for user in context_users:
            user_id = user.get("user_id")
            if not isinstance(user_id, int) or user_id not in rating_by_user:
                continue
            label = (
                str(user.get("display_name") or "").strip()
                or str(user.get("name") or "").strip()
                or f"user_{user_id}"
            )
            entries.append(f"{label}={rating_by_user[user_id]}")

        if not entries:
            return ""
        return (
            "Rating reference only (not part of the conversation, do not quote it unless directly asked): "
            + ", ".join(entries)
        )

    def _social_message_excerpt(
        self, message: dict[str, Any], max_chars: int = 140
    ) -> str:
        text = self._truncate_fenced_code(
            str(message.get("content") or "").strip() or "[no text]"
        )
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    def _build_social_context_users(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        context_users: dict[str, dict[str, Any]] = {}
        for message in messages:
            if bool(message.get("author_is_bot")):
                continue
            user_key = str(
                message.get("author_id")
                or message.get("author_handle")
                or message.get("author_name")
                or ""
            )
            if not user_key:
                continue
            existing = context_users.get(user_key, {})
            author_name = str(message.get("author_name") or "").strip()
            author_display_name = str(message.get("author_display_name") or "").strip()
            author_handle = str(message.get("author_handle") or "").strip()
            if author_name:
                existing["name"] = author_name
            if author_display_name:
                existing["display_name"] = author_display_name
            if author_handle:
                existing["handle"] = author_handle
            existing["user_id"] = user_key
            context_users[user_key] = existing
        return list(context_users.values())

    def _render_social_message_payloads(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        remaining_tokens = self.available_tokens("")
        for message in reversed(messages):
            payload_tokens = self.count_tokens(
                json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            )
            if remaining_tokens - payload_tokens < 0:
                logger.warning("Not enough tokens available for the social message payload.")
                continue
            remaining_tokens -= payload_tokens
            payloads.append(message)
        return list(reversed(payloads))

    def _build_social_mention_payload(
        self,
        *,
        platform: str,
        account_handle: str,
        question: str,
        messages: list[dict[str, Any]],
        current_message: dict[str, Any],
        max_chars: int,
        reply_mode: Optional[str] = None,
        retry_hint: Optional[str] = None,
    ) -> str:
        resolved_reply_mode = reply_mode or self._classify_mention_reply_mode(
            question,
            has_reference=bool(current_message.get("reply_to_message_id")),
            retry_hint=retry_hint,
        )
        merged_messages = list(messages)
        if not merged_messages or merged_messages[-1].get("id") != current_message.get("id"):
            merged_messages.append(current_message)
        rendered_messages = self._render_social_message_payloads(merged_messages)
        context_users = self._build_social_context_users(merged_messages)

        reply_chain_payload = None
        reply_chain_tail = ""
        if current_message.get("reply_to_message_id") and rendered_messages:
            parent_message = rendered_messages[-2] if len(rendered_messages) >= 2 else None
            reply_chain_payload = {
                "focus": "primary",
                "reason": "This is a reply-chain turn. Prioritize the thread over isolated posts.",
                "messages": rendered_messages,
            }
            tail_lines = [
                "[Reply chain focus]",
                "- this reply chain is the primary context",
            ]
            if parent_message is not None:
                parent_label = (
                    str(parent_message.get("author_display_name") or "").strip()
                    or str(parent_message.get("author_name") or "").strip()
                    or "unknown"
                )
                tail_lines.append(
                    f"- direct parent from {parent_label}: {self._social_message_excerpt(parent_message)}"
                )
            current_label = (
                str(current_message.get("author_display_name") or "").strip()
                or str(current_message.get("author_name") or "").strip()
                or "unknown"
            )
            tail_lines.append(
                f"- current message from {current_label}: {self._social_message_excerpt(current_message)}"
            )
            reply_chain_tail = "\n".join(tail_lines)

        payload_body = self._json_dumps(
            {
                "environment": {
                    "platform": platform,
                    "bot_name": "Sigrok",
                    "account_handle": account_handle,
                },
                "event": {
                    "type": "mention_reply",
                    "question": question,
                    "reply_mode": resolved_reply_mode,
                    "current_message": current_message,
                    "retry_hint": retry_hint,
                },
                "context_users": context_users,
                "messages": rendered_messages,
                "reply_chain": reply_chain_payload,
                "response": {
                    "kind": f"{platform}_reply",
                    "plain_text_only": True,
                    "max_chars": max_chars,
                },
            }
        )
        payload_parts = [payload_body]
        if reply_chain_tail:
            payload_parts.append(reply_chain_tail)
        return "\n\n".join(part for part in payload_parts if part)

    def _message_excerpt(self, message: Message, max_chars: int = 140) -> str:
        text = self._message_content(message)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    @staticmethod
    def _infer_reply_question_scope(question: str) -> str:
        lowered = question.lower()
        if any(word in lowered for word in ("image", "picture", "photo", "screenshot", "meme")):
            return "describe_image"
        if any(word in lowered for word in ("person", "guy", "girl", "man", "woman", "he ", "she ")):
            return "describe_person"
        if any(word in lowered for word in ("funny", "unfunny", "joke", "humor", "humour")):
            return "judge_humor"
        return "reply_to_latest_message"

    def _find_image_source_message(
        self, current_message: Message, ancestors: list[Message]
    ) -> Optional[Message]:
        if self._message_has_images(current_message):
            return current_message
        for candidate in reversed(ancestors):
            if self._message_has_images(candidate):
                return candidate
        return None

    def _message_focus_payload(
        self,
        message: Message,
        image_indexes: Optional[dict[tuple[int, int], int]] = None,
    ) -> dict[str, Any]:
        payload = self._message_payload(message, image_indexes)
        return {
            "id": payload["id"],
            "author_id": payload["author_id"],
            "author_name": payload["author_name"],
            "author_display_name": payload["author_display_name"],
            "author_is_bot": payload["author_is_bot"],
            "reply_to_message_id": payload["reply_to_message_id"],
            "content": payload["content"],
            "content_excerpt": self._message_excerpt(message),
            "has_images": self._message_has_images(message),
            "attachments": payload["attachments"],
        }

    def _surrounding_image_payload(
        self,
        message: Message,
        image_indexes: Optional[dict[tuple[int, int], int]] = None,
    ) -> Optional[dict[str, Any]]:
        if image_indexes is None:
            return None

        images: list[dict[str, Any]] = []
        for attachment_idx, attachment in enumerate(message.attachments):
            vision_input_index = image_indexes.get((message.id, attachment_idx))
            if vision_input_index is None:
                continue
            images.append(
                {
                    "filename": attachment.filename,
                    "vision_input_index": vision_input_index,
                }
            )

        for embed_idx, embed in enumerate(message.embeds):
            vision_input_index = image_indexes.get((message.id, len(message.attachments) + embed_idx))
            if vision_input_index is None:
                continue
            images.append(
                {
                    "filename": getattr(embed, "title", None)
                    or getattr(embed, "url", None)
                    or "embed_image",
                    "vision_input_index": vision_input_index,
                }
            )

        if not images:
            return None

        return {
            "message_id": message.id,
            "speaker": getattr(message.author, "display_name", None) or self._speaker_label(message),
            "content_excerpt": self._message_excerpt(message),
            "images": images,
        }

    def _render_surrounding_image_payloads(
        self,
        messages: list[Message],
        image_indexes: Optional[dict[tuple[int, int], int]] = None,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        remaining_tokens = self.available_tokens("")
        for message in reversed(messages):
            payload = self._surrounding_image_payload(message, image_indexes)
            if payload is None:
                continue
            payload_tokens = self.count_tokens(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            if remaining_tokens - payload_tokens < 0:
                logger.warning("Not enough tokens available for the surrounding image payload.")
                continue
            remaining_tokens -= payload_tokens
            payloads.append(payload)
        return list(reversed(payloads))

    async def _build_reply_chain_focus(
        self,
        source_message: Message,
        bot_user_id: Optional[int],
        user_ids: Optional[set[int]],
        image_indexes: Optional[dict[tuple[int, int], int]] = None,
        *,
        question: Optional[str] = None,
    ) -> tuple[list[Message], Optional[dict[str, Any]], str]:
        if not source_message.reference or source_message.reference.message_id is None:
            return [], None, ""

        parent_message = await self._resolve_reference_message(source_message)
        if parent_message is None:
            return [], None, ""
        if not self._eligible_for_transcript(
            parent_message, bot_user_id, user_ids, skip_user_id_filter=True
        ):
            return [], None, ""

        chain_messages = [parent_message]
        if not source_message.author.bot:
            chain_messages.append(source_message)

        current_message = source_message
        image_source_message = self._find_image_source_message(
            current_message, [parent_message]
        )
        question_scope = self._infer_reply_question_scope(question or current_message.content or "")
        payload = {
            "focus": "primary",
            "reason": "This is a reply-chain turn. Answer the latest message using only the direct parent reply target.",
            "question_scope": question_scope,
            "current_message": self._message_focus_payload(current_message, image_indexes),
            "direct_parent_message": self._message_focus_payload(parent_message, image_indexes),
            "image_source_message": (
                self._message_focus_payload(image_source_message, image_indexes)
                if image_source_message is not None
                else None
            ),
            "reply_chain_transcript": self._render_plain_context_messages(chain_messages),
        }
        tail_lines = [
            "[Reply chain focus]",
            "- this reply chain is the primary context",
            f"- current message from {self._speaker_label(current_message)}: {self._message_excerpt(current_message)}",
            f"- direct parent from {self._speaker_label(parent_message)}: {self._message_excerpt(parent_message)}",
        ]
        if image_source_message is not None:
            tail_lines.append(
                f"- primary visual source from {self._speaker_label(image_source_message)}: "
                f"{self._message_excerpt(image_source_message)}"
            )
        tail_lines.append(f"- question_scope: {question_scope}")
        tail = "\n".join(tail_lines)
        return chain_messages, payload, tail

    def count_tokens(self, input: str) -> int:
        if self.tokenizer is None:
            return max(1, len(input) // 4)
        return len(self.tokenizer.encode(input))  # type: ignore

    def available_tokens(self, input: str) -> int:
        return (
            self.settings.genai.tokens.limit
            - self.settings.genai.tokens.overhead_max
            - self.settings.genai.tokens.prompt_max
            - self.settings.genai.tokens.output_max
            - self.count_tokens(input)
        )

    @classmethod
    def _question_keywords(cls, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {
            word
            for word in words
            if len(word) >= 3 and word not in cls._QUESTION_STOP_WORDS
        }

    @classmethod
    def _classify_mention_reply_mode(
        cls,
        question: str,
        *,
        has_reference: bool = False,
        retry_hint: Optional[str] = None,
    ) -> str:
        if retry_hint:
            return "repair"
        if has_reference:
            return "reply_chain"

        lowered = question.strip().lower()
        word_count = len(re.findall(r"\b\w+\b", lowered))
        keywords = cls._question_keywords(lowered)
        ambiguous_followups = {
            "more than me",
            "what do you mean",
            "how so",
            "why",
            "why?",
            "and?",
            "so?",
        }
        casual_prompts = {
            "you good",
            "you good?",
            "you there",
            "you there?",
            "tea or coffee",
            "coffee or tea",
            "sup",
            "hello?",
            "yo",
            "hey",
        }

        if lowered in ambiguous_followups:
            return "ambiguous_followup"
        if lowered in casual_prompts:
            return "smalltalk"
        if word_count <= 3 and not keywords:
            return "smalltalk"
        if word_count <= 4 and len(keywords) <= 2:
            return "quick_question"
        if word_count >= 18 or len(question) >= 140:
            return "serious_discussion"
        if question.strip().endswith("?"):
            return "direct_question"
        return "discussion"

    @classmethod
    def _mention_context_limit(
        cls,
        question: str,
        default_limit: int,
        *,
        has_reference: bool = False,
        retry_hint: Optional[str] = None,
    ) -> int:
        mode = cls._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        if mode == "reply_chain":
            return min(default_limit, 2)
        if mode in {"smalltalk", "repair"}:
            return min(default_limit, 2)
        if mode in {"quick_question", "ambiguous_followup"}:
            return min(default_limit, 3)
        return default_limit

    @classmethod
    def _mention_length_hint(cls, reply_mode: str) -> str:
        if reply_mode == "reply_chain":
            return "prefer 1 to 2 sentences that continue the immediate thread"
        if reply_mode in {"smalltalk", "quick_question"}:
            return "prefer 1 short sentence"
        if reply_mode == "ambiguous_followup":
            return "ask 1 short clarifying question"
        if reply_mode == "repair":
            return "keep it brief and answer only the latest message"
        if reply_mode == "serious_discussion":
            return "prefer 2 to 4 sentences"
        return "prefer 1 to 3 sentences"

    _CODE_FENCE_TRUNC_LINES = 10

    @classmethod
    def _truncate_fenced_code(cls, content: str) -> str:
        """Truncate fenced ``` blocks so long HTML/code does not dominate context."""
        if "```" not in content:
            return content
        out: list[str] = []
        i = 0
        while i < len(content):
            start = content.find("```", i)
            if start < 0:
                out.append(content[i:])
                break
            out.append(content[i:start])
            end = content.find("```", start + 3)
            if end < 0:
                out.append(content[start:])
                break
            inner = content[start + 3 : end]
            lines = inner.split("\n")
            lang_prefix = ""
            body_lines = lines
            if lines:
                head = lines[0].strip()
                if head and len(head) <= 20 and " " not in head and "\t" not in head:
                    lang_prefix = lines[0]
                    body_lines = lines[1:]
            if len(body_lines) > cls._CODE_FENCE_TRUNC_LINES:
                keep = body_lines[: cls._CODE_FENCE_TRUNC_LINES]
                omitted = len(body_lines) - cls._CODE_FENCE_TRUNC_LINES
                body = "\n".join(keep) + f"\n… [{omitted} lines truncated for context]"
            else:
                body = "\n".join(body_lines)
            if lang_prefix:
                rebuilt = f"```{lang_prefix}\n{body}\n```"
            else:
                rebuilt = f"```{body}\n```" if body else "```\n```"
            out.append(rebuilt)
            i = end + 3
        return "".join(out)

    def _speaker_label(self, message: Message) -> str:
        if message.author.bot:
            return "Sigrok"
        return message.author.name

    def _message_content(self, message: Message) -> str:
        content = (message.content or "").strip() or "[no text]"
        return self._truncate_fenced_code(content)

    def _is_supported_image_attachment(self, attachment: Any) -> bool:
        content_type = (getattr(attachment, "content_type", None) or "").lower()
        if content_type.startswith("image/"):
            return True
        filename = (getattr(attachment, "filename", "") or "").lower()
        return any(filename.endswith(ext) for ext in self._SUPPORTED_IMAGE_EXTENSIONS)

    _IMAGE_MAGIC = {
        b"\x89PNG\r\n\x1a\n": "image/png",
        b"\xff\xd8": "image/jpeg",
        b"GIF8": "image/gif",
    }

    @staticmethod
    def _is_valid_image_bytes(data: bytes) -> bool:
        if len(data) < 8:
            return False
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return True
        if data[:2] == b"\xff\xd8":
            return True
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return True
        if data[:4] == b"GIF8":
            return True
        if data[:2] == b"BM":
            return True
        return False

    @staticmethod
    def _message_has_images(message: Message) -> bool:
        if message.attachments and any(
            (a.content_type or "").startswith("image/") for a in message.attachments
        ):
            return True
        for embed in message.embeds:
            if (embed.image and embed.image.url) or (embed.thumbnail and embed.thumbnail.url):
                return True
        return False

    async def _download_image_url(self, url: str) -> Optional[bytes]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.read()
                    if len(data) > self._MAX_INLINE_IMAGE_BYTES:
                        return None
                    return data
        except Exception as exc:
            logger.warning(f"Failed to download embed image {url}: {exc}")
            return None

    def _process_image_data(
        self, data: bytes, source: str, *, max_edge: int = 768
    ) -> Optional[str]:
        if not self._is_valid_image_bytes(data):
            logger.warning(f"Skipping non-image data from {source} (magic: {data[:4]!r}).")
            return None
        try:
            img = PILImage.open(io.BytesIO(data))
            if max(img.size) > max_edge:
                img.thumbnail((max_edge, max_edge), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.warning(f"Failed to process image from {source}: {exc}")
            return None

    async def _collect_inline_images(
        self,
        messages: list[Message],
        *,
        high_detail_message_ids: Optional[set[int]] = None,
    ) -> tuple[dict[tuple[int, int], int], list[str]]:
        if not self._VISION_ENABLED:
            return {}, []

        image_indexes: dict[tuple[int, int], int] = {}
        inline_images: list[str] = []
        high_detail_message_ids = high_detail_message_ids or set()
        low_detail_max_edge = 96

        for message in messages:
            max_edge = 768 if message.id in high_detail_message_ids else low_detail_max_edge
            for attachment_idx, attachment in enumerate(message.attachments):
                if len(inline_images) >= self._MAX_INLINE_IMAGES:
                    return image_indexes, inline_images
                if not self._is_supported_image_attachment(attachment):
                    continue
                size = getattr(attachment, "size", 0) or 0
                if size > self._MAX_INLINE_IMAGE_BYTES:
                    continue
                try:
                    data = await attachment.read(use_cached=True)
                except Exception:
                    continue
                b64 = self._process_image_data(
                    data, attachment.filename, max_edge=max_edge
                )
                if b64:
                    inline_images.append(b64)
                    image_indexes[(message.id, attachment_idx)] = len(inline_images) - 1

            for embed_idx, embed in enumerate(message.embeds):
                if len(inline_images) >= self._MAX_INLINE_IMAGES:
                    return image_indexes, inline_images
                img_url = None
                if embed.image and embed.image.url:
                    img_url = embed.image.url
                elif embed.thumbnail and embed.thumbnail.url:
                    img_url = embed.thumbnail.url
                if not img_url:
                    continue
                data = await self._download_image_url(img_url)
                if not data:
                    continue
                b64 = self._process_image_data(data, img_url, max_edge=max_edge)
                if b64:
                    inline_images.append(b64)
                    image_indexes[(message.id, len(message.attachments) + embed_idx)] = len(inline_images) - 1

        return image_indexes, inline_images

    def _speaker_label_with_rating(
        self, message: Message, rating_by_user: Optional[dict[int, int]] = None
    ) -> str:
        speaker = self._speaker_label(message)
        if rating_by_user is None or message.author.bot:
            return speaker
        r = rating_by_user.get(message.author.id)
        if r is None:
            return speaker
        return f"{speaker} (rating: {r})"

    def _message_payload(
        self,
        message: Message,
        image_indexes: Optional[dict[tuple[int, int], int]] = None,
        rating_by_user: Optional[dict[int, int]] = None,
    ) -> dict[str, Any]:
        attachments = []
        for attachment_idx, attachment in enumerate(message.attachments):
            attachment_payload: dict[str, Any] = {
                "filename": attachment.filename,
                "content_type": getattr(attachment, "content_type", None),
                "size": getattr(attachment, "size", None),
                "url": getattr(attachment, "url", None),
                "is_image": self._is_supported_image_attachment(attachment),
            }
            if image_indexes is not None:
                vision_input_index = image_indexes.get((message.id, attachment_idx))
                if vision_input_index is not None:
                    attachment_payload["vision_input_index"] = vision_input_index
            attachments.append(attachment_payload)
        display_name = getattr(message.author, "display_name", None)
        author_rating = None
        if rating_by_user is not None and not message.author.bot:
            author_rating = rating_by_user.get(message.author.id)
        return {
            "id": message.id,
            "author_id": message.author.id,
            "author_name": self._speaker_label(message),
            "author_display_name": display_name,
            "author_rating": author_rating,
            "author_is_bot": bool(message.author.bot),
            "created_at": message.created_at.isoformat(),
            "reply_to_message_id": (
                message.reference.message_id
                if message.reference and message.reference.message_id is not None
                else None
            ),
            "content": self._message_content(message),
            "attachments": attachments,
        }

    def format_message(
        self, message: Message, rating_by_user: Optional[dict[int, int]] = None
    ) -> str:
        content = self._message_content(message)
        if message.attachments:
            attachment_names = ", ".join(
                attachment.filename for attachment in message.attachments
            )
            content = f"{content} [attachments: {attachment_names}]"
        speaker = self._speaker_label_with_rating(message, rating_by_user)
        return f"{speaker}: {content}"

    def _format_plain_context_message(self, message: Message) -> str:
        content = self._message_content(message)
        if message.attachments:
            attachment_names = ", ".join(
                attachment.filename for attachment in message.attachments
            )
            content = f"{content} [attachments: {attachment_names}]"
        speaker = getattr(message.author, "display_name", None) or self._speaker_label(message)
        return f"{speaker}: {content}"

    def _render_messages(
        self, messages: list[Message], rating_by_user: Optional[dict[int, int]] = None
    ) -> str:
        lines: list[str] = []
        remaining_tokens = self.available_tokens("")
        for message in reversed(messages):
            formatted_message = self.format_message(message, rating_by_user)
            message_tokens = self.count_tokens(formatted_message)
            if remaining_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the message.")
                continue
            remaining_tokens -= message_tokens
            lines.append(formatted_message)
        return "\n".join(reversed(lines))

    def _render_plain_context_messages(self, messages: list[Message]) -> str:
        lines: list[str] = []
        remaining_tokens = self.available_tokens("")
        for message in reversed(messages):
            formatted_message = self._format_plain_context_message(message)
            message_tokens = self.count_tokens(formatted_message)
            if remaining_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the plain context message.")
                continue
            remaining_tokens -= message_tokens
            lines.append(formatted_message)
        return "\n".join(reversed(lines))

    def _render_message_payloads(
        self,
        messages: list[Message],
        image_indexes: Optional[dict[tuple[int, int], int]] = None,
        rating_by_user: Optional[dict[int, int]] = None,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        remaining_tokens = self.available_tokens("")
        for message in reversed(messages):
            payload = self._message_payload(message, image_indexes, rating_by_user)
            payload_tokens = self.count_tokens(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            if remaining_tokens - payload_tokens < 0:
                logger.warning("Not enough tokens available for the message payload.")
                continue
            remaining_tokens -= payload_tokens
            payloads.append(payload)
        return list(reversed(payloads))

    async def _collect_history_messages(
        self,
        channel: Any,
        *,
        before: datetime,
        after: datetime,
        limit: int,
        include_bots: bool = False,
    ) -> list[Message]:
        messages: list[Message] = []
        context_tokens = self.available_tokens("")
        async for message in channel.history(
            before=before,
            after=after,
            limit=limit,
            oldest_first=False,
        ):
            if message.author.bot and not include_bots:
                continue

            formatted_message = self.format_message(message)
            message_tokens = self.count_tokens(formatted_message)

            if context_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the message.")
                break

            context_tokens -= message_tokens
            messages.append(message)

        return list(reversed(messages))

    def _eligible_for_transcript(
        self,
        message: Message,
        bot_user_id: Optional[int],
        user_ids: Optional[set[int]],
        *,
        skip_user_id_filter: bool = False,
    ) -> bool:
        """Whether a message may appear in the recent transcript."""
        if message.author.bot and (bot_user_id is None or message.author.id != bot_user_id):
            return False
        if (
            not skip_user_id_filter
            and user_ids
            and (not message.author.bot)
            and message.author.id not in user_ids
        ):
            return False
        return True

    async def _resolve_reference_message(self, message: Message) -> Optional[Message]:
        if not message.reference or message.reference.message_id is None:
            return None
        if message.reference.resolved and isinstance(message.reference.resolved, Message):
            return message.reference.resolved
        try:
            return await message.channel.fetch_message(message.reference.message_id)
        except Exception as exc:
            logger.debug(f"Unable to resolve reply reference {message.reference.message_id}: {exc}")
            return None

    _MAX_REPLY_CHAIN_DEPTH = 6

    async def _collect_reference_chain_messages(
        self,
        source_message: Message,
        bot_user_id: Optional[int],
        user_ids: Optional[set[int]] = None,
    ) -> list[Message]:
        """Walk the reply chain up to _MAX_REPLY_CHAIN_DEPTH eligible ancestors."""
        chain: list[Message] = []
        seen_ids: set[int] = {source_message.id}
        current = source_message
        hops = 0

        while len(chain) < self._MAX_REPLY_CHAIN_DEPTH and hops < self._MAX_REPLY_CHAIN_DEPTH * 3:
            ref = await self._resolve_reference_message(current)
            if ref is None or ref.id in seen_ids:
                break
            seen_ids.add(ref.id)
            hops += 1
            if self._eligible_for_transcript(
                ref, bot_user_id, user_ids, skip_user_id_filter=True
            ):
                chain.append(ref)
            current = ref

        chain.reverse()
        return chain

    async def build_recent_context_for_message(
        self,
        source_message: Message,
        limit: int,
        user_ids: Optional[set[int]] = None,
        include_current: bool = False,
    ) -> str:
        return self._render_messages(
            await self._collect_recent_context_messages(
                source_message,
                limit,
                user_ids=user_ids,
                include_current=include_current,
            )
        )

    async def _collect_recent_context_messages(
        self,
        source_message: Message,
        limit: int,
        user_ids: Optional[set[int]] = None,
        include_current: bool = False,
        history_before: Optional[datetime] = None,
        *,
        merge_reply_chain: bool = True,
    ) -> list[Message]:
        """Build transcript: last `limit` human turns plus interleaved Sigrok replies (chronological)."""
        guild = source_message.guild
        bot_user_id: Optional[int] = guild.me.id if guild and guild.me else None

        raw: list[Message] = []
        history_limit = max(limit * 3, 20)
        before_anchor = (
            history_before if history_before is not None else source_message.created_at
        )
        async for message in source_message.channel.history(
            before=before_anchor,
            limit=history_limit,
            oldest_first=False,
        ):
            raw.append(message)
        raw.reverse()

        eligible: list[Message] = []
        for message in raw:
            if not self._eligible_for_transcript(message, bot_user_id, user_ids):
                continue
            eligible.append(message)

        window: list[Message] = []
        humans = 0
        for message in reversed(eligible):
            window.insert(0, message)
            if not message.author.bot:
                humans += 1
                if humans >= limit:
                    break

        by_id: dict[int, Message] = {m.id: m for m in window}

        # User hit Reply on a message — include the full ancestor chain so "this"
        # / "that" stays grounded even in long Discord reply threads.
        if merge_reply_chain:
            for ref in await self._collect_reference_chain_messages(
                source_message, bot_user_id, user_ids
            ):
                if ref.id not in by_id:
                    by_id[ref.id] = ref

        merged = sorted(by_id.values(), key=lambda m: (m.created_at, m.id))

        if include_current and not source_message.author.bot:
            merged.append(source_message)

        return merged

    async def read_context(self, ctx: ApplicationContext | Reaction | Message) -> str:
        if isinstance(ctx, ApplicationContext):
            channel = ctx.channel
        elif isinstance(ctx, Message):
            channel = ctx.channel
        elif isinstance(ctx, Reaction):
            channel = ctx.message.channel
        else:
            logger.error("Invalid context type provided.")
            return ""

        messages = await self._collect_history_messages(
            channel,
            before=datetime.now(),
            after=datetime.now() - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
        )
        return self._render_messages(messages)

    async def read_current_context(self, ctx: ApplicationContext) -> str:
        messages = await self._collect_history_messages(
            ctx.channel,
            before=datetime.now(),
            after=datetime.now() - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
        )
        return self._render_messages(messages)

    async def read_message_context(self, msg: Message) -> str:
        messages = await self._collect_history_messages(
            msg.channel,
            before=msg.created_at,
            after=msg.created_at - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
        )
        return self._render_messages(messages)

    async def read_reaction_context(self, reaction: Reaction) -> str:
        messages = await self._collect_history_messages(
            reaction.message.channel,
            before=reaction.message.created_at,
            after=reaction.message.created_at
            - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
        )
        return self._render_messages(messages)

    def _build_environment_payload(
        self, ctx: ApplicationContext | Reaction | Message
    ) -> dict[str, Any]:
        if isinstance(ctx, Reaction):
            guild = ctx.message.guild
            channel = ctx.message.channel
        elif isinstance(ctx, Message):
            guild = ctx.guild
            channel = ctx.channel
        else:
            guild = ctx.guild
            channel = ctx.channel

        return {
            "platform": "discord",
            "bot_name": "Sigrok",
            "guild_id": getattr(guild, "id", None),
            "guild_name": getattr(guild, "name", None),
            "channel_id": getattr(channel, "id", None),
            "channel_name": getattr(channel, "name", None),
        }

    async def _build_event_payload(
        self,
        ctx: ApplicationContext | Reaction | Message,
        *,
        event_type: str,
        messages: list[Message],
        event: dict[str, Any],
        response: dict[str, Any],
        context_users: Optional[list[dict[str, Any]]] = None,
        rating_by_user: Optional[dict[int, int]] = None,
    ) -> tuple[str, list[str]]:
        image_indexes, inline_images = await self._collect_inline_images(messages)
        return self._json_dumps(
            {
                "environment": self._build_environment_payload(ctx),
                "event": {"type": event_type, **event},
                "context_users": context_users or self._build_context_users(messages),
                "messages": self._render_message_payloads(
                    messages, image_indexes, rating_by_user
                ),
                "response": response,
            }
        ), inline_images

    async def _build_mention_reply_payload(
        self,
        message: Message,
        messages: list[Message],
        question: str,
        user_ids: Optional[set[int]] = None,
        *,
        reply_mode: Optional[str] = None,
        retry_hint: Optional[str] = None,
        skip_images: bool = False,
    ) -> tuple[str, list[str], str]:
        image_source_messages = list(messages)
        if all(existing.id != message.id for existing in image_source_messages):
            image_source_messages.append(message)
        bot_user_id: Optional[int] = (
            message.guild.me.id if message.guild and message.guild.me else None
        )
        if skip_images:
            ref = None
            image_indexes: dict[tuple[int, int], int] = {}
            inline_images: list[str] = []
        else:
            ref = await self._resolve_reference_message(message)
            if (
                ref is not None
                and self._message_has_images(ref)
                and ref.id not in {m.id for m in image_source_messages}
            ):
                image_source_messages.append(ref)
            high_detail_message_ids = {message.id}
            if ref is not None:
                high_detail_message_ids.add(ref.id)
            image_indexes, inline_images = await self._collect_inline_images(
                image_source_messages,
                high_detail_message_ids=high_detail_message_ids,
            )
        resolved_reply_mode = reply_mode or self._classify_mention_reply_mode(
            question,
            has_reference=bool(message.reference and message.reference.message_id),
            retry_hint=retry_hint,
        )
        chain_messages, reply_chain_payload, reply_chain_tail = await self._build_reply_chain_focus(
            message,
            bot_user_id,
            user_ids,
            image_indexes,
            question=question,
        )
        if resolved_reply_mode == "reply_chain" and chain_messages:
            focus_messages = list(chain_messages)
        else:
            focus_messages = [message]
        focus_message_ids = {m.id for m in focus_messages}
        surrounding_messages = [m for m in messages if m.id not in focus_message_ids]
        context_source_messages = list(focus_messages)
        ambient_messages: list[Message] = []
        surrounding_transcript = self._render_plain_context_messages(surrounding_messages)
        surrounding_image_context = self._render_surrounding_image_payloads(
            surrounding_messages, image_indexes
        )
        rating_by_user = await self._build_guild_rating_map(message)
        context_users = self._build_context_users(context_source_messages)
        rating_reference_note = self._format_rating_reference_note(
            context_users, rating_by_user
        )
        payload_body = self._json_dumps(
            {
                "environment": self._build_environment_payload(message),
                "instructions": [
                    "Answer current_message. surrounding_transcript is background only.",
                    "Need fresh info or sources? Call search_web/fetch_url first. Never invent URLs.",
                ],
                "event": {
                    "type": "mention_reply",
                    "question": question,
                    "reply_mode": resolved_reply_mode,
                    "current_message_id": message.id,
                    "target_user_ids": sorted(user_ids) if user_ids else [],
                    "retry_hint": retry_hint,
                },
                "context_users": context_users,
                "current_message": self._message_focus_payload(message, image_indexes),
                "messages": self._render_message_payloads(ambient_messages, image_indexes),
                "surrounding_transcript": surrounding_transcript,
                "surrounding_image_context": surrounding_image_context,
                "thread_focus": reply_chain_payload,
                "response": {
                    "kind": "discord_message",
                    "plain_text_only": True,
                    "max_chars": 2000,
                },
            }
        )
        payload_parts = [payload_body]
        if reply_chain_tail:
            payload_parts.append(reply_chain_tail)
        if rating_reference_note:
            payload_parts.insert(0, rating_reference_note)
        payload = "\n\n".join(part for part in payload_parts if part)
        return payload, inline_images, resolved_reply_mode

    async def _build_guild_rating_map_from_guild(self, guild: Any) -> dict[int, int]:
        if guild is None:
            return {}
        users = await db.read_present_users(guild.id)
        return {
            user.user_id: (user.rating if user.rating is not None else 100)
            for user in users
        }

    async def _build_guild_rating_map(self, message: Message) -> dict[int, int]:
        return await self._build_guild_rating_map_from_guild(message.guild)

    async def _build_debate_payload(
        self,
        ctx: ApplicationContext | Reaction,
        messages: list[Message],
        participants: list[str],
        *,
        topic: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        image_source_messages = list(messages)
        if isinstance(ctx, Reaction) and all(
            existing.id != ctx.message.id for existing in image_source_messages
        ):
            image_source_messages.append(ctx.message)
        guild = ctx.message.guild if isinstance(ctx, Reaction) else ctx.guild
        rating_by_user = await self._build_guild_rating_map_from_guild(guild)
        context_users = self._build_context_users(image_source_messages, rating_by_user)
        image_indexes, inline_images = await self._collect_inline_images(
            image_source_messages
        )
        payload = self._json_dumps(
            {
                "environment": self._build_environment_payload(ctx),
                "event": {
                    "type": "debate_resolution",
                    "participants": participants,
                    "topic": topic,
                    "judgement": {
                        "priority": [
                            "soundness_and_logical_validity",
                            "internal_consistency",
                            "rhetorical_effectiveness",
                        ],
                        "allow_general_knowledge": True,
                        "transcript_claims_must_match_messages": True,
                    },
                    "trigger_message": (
                        self._message_payload(ctx.message, image_indexes, rating_by_user)
                        if isinstance(ctx, Reaction)
                        else None
                    ),
                },
                "context_users": context_users,
                "messages": self._render_message_payloads(
                    messages, image_indexes, rating_by_user
                ),
                "response": {
                    "kind": "json",
                    "schema": {
                        "winner": participants + ["draw", "none"],
                        "reason": "short in-character explanation",
                    },
                },
            }
        )
        return payload, inline_images

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict[str, Any]]:
        text = response.strip()
        candidates = [text]

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            candidates.insert(0, fenced_match.group(1))

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _parse_debate_response(
        self, response: str, participants: list[str]
    ) -> tuple[str, str]:
        parsed = self._parse_json_response(response)
        winner = ""
        reason = response.strip()

        if parsed is not None:
            winner = str(parsed.get("winner", "")).strip()
            reason = str(parsed.get("reason") or reason).strip()

        if not winner:
            match = re.search(r"winner:\s*([^\n*]+)", response, flags=re.IGNORECASE)
            if match is not None:
                winner = match.group(1).strip()

        allowed = {name.lower(): name for name in participants}
        lowered = winner.lower()
        if lowered in allowed:
            normalized_winner = allowed[lowered]
        elif lowered in {"draw", "none"}:
            normalized_winner = lowered
        else:
            normalized_winner = "error"

        if not reason:
            reason = "No reasoning provided."

        visible_winner = (
            normalized_winner if normalized_winner != "error" else "error"
        )
        public_response = f"**Winner: {visible_winner}**\n{reason}".strip()
        return normalized_winner, public_response[:1999]

    async def answer_message_question(
        self,
        message: Message,
        question: str,
        user_ids: Optional[set[int]] = None,
        retry_hint: Optional[str] = None,
        history_before: Optional[datetime] = None,
        *,
        recent_context_human_turns: Optional[int] = None,
        merge_reply_chain: bool = True,
    ) -> str:
        raise NotImplementedError("answer_message_question must be implemented by subclasses")

    async def answer_social_question(
        self,
        *,
        platform: str,
        account_handle: str,
        question: str,
        messages: list[dict[str, Any]],
        current_message: dict[str, Any],
        max_chars: int,
        retry_hint: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("answer_social_question must be implemented by subclasses")

    async def judge_debate(
        self,
        ctx: ApplicationContext | Reaction,
        participants: list[str],
        topic: Optional[str] = None,
    ) -> tuple[str, str]:
        raise NotImplementedError("judge_debate must be implemented by subclasses")


class GenAIGpt(GenAIBase):
    client: OpenAI
    tokenizer: Tokenizer

    def __init__(self, settings: Settings) -> None:
        assert settings.genai.model.startswith("gpt"), "GenAIGpt requires a GPT model"
        self.settings = settings
        self.client = OpenAI(api_key=settings.tokens.gpt)
        self.tokenizer = Tokenizer.from_pretrained("Xenova/gpt-4o")

    def _request_completion(self, messages: list[ChatMessage]) -> str:
        response = self.client.chat.completions.create(
            model=settings.genai.model,
            messages=[
                {"role": message.role, "content": message.content} for message in messages
            ],  # type: ignore
            max_tokens=settings.genai.tokens.output_max,
        )
        content = response.choices[0].message.content
        logger.info(f"GPT response: {content}")
        return content if content else "No response from GPT"

    async def _build_prompt(self, user_payload: str) -> list[ChatMessage]:
        messages = []
        messages.append(
            ChatMessage(
                role=Role.USER,
                content=user_payload,
            )
        )
        return messages

    async def judge_debate(
        self,
        ctx: ApplicationContext | Reaction,
        participants: list[str],
        topic: Optional[str] = None,
    ) -> tuple[str, str]:
        messages_for_context = await self._collect_history_messages(
            ctx.message.channel if isinstance(ctx, Reaction) else ctx.channel,
            before=datetime.now(),
            after=datetime.now() - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
        )

        if not messages_for_context:
            logger.warning("No conversation history found.")
            return "error", "No conversation history available to judge."

        user_payload, inline_images = await self._build_debate_payload(
            ctx, messages_for_context, participants, topic=topic
        )
        try:
            response = self._request_completion(
                [
                    ChatMessage(
                        role=Role.SYSTEM,
                        content=self._build_personality_system_prompt("discord"),
                    ),
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                        images=inline_images or None,
                    ),
                ]
            )
            return self._parse_debate_response(response, participants)
        except Exception as exc:
            logger.error(f"Error occurred in judge_debate: {exc}")
            return "error", "An error occurred while judging the debate."

    async def answer_message_question(
        self,
        message: Message,
        question: str,
        user_ids: Optional[set[int]] = None,
        retry_hint: Optional[str] = None,
        history_before: Optional[datetime] = None,
        *,
        recent_context_human_turns: Optional[int] = None,
        merge_reply_chain: bool = True,
    ) -> str:
        has_reference = bool(message.reference and message.reference.message_id)
        reply_mode = self._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        _ctx_limit = (
            recent_context_human_turns
            if recent_context_human_turns is not None
            else self._mention_context_limit(
                question,
                settings.genai.question.recent_messages,
                has_reference=has_reference,
                retry_hint=retry_hint,
            )
        )
        messages_for_context = await self._collect_recent_context_messages(
            message,
            _ctx_limit,
            user_ids=user_ids,
            include_current=has_reference,
            history_before=history_before,
            merge_reply_chain=merge_reply_chain,
        )
        user_payload, inline_images, resolved_reply_mode = await self._build_mention_reply_payload(
            message,
            messages_for_context,
            question,
            user_ids=user_ids,
            reply_mode=reply_mode,
            retry_hint=retry_hint,
        )
        try:
            response = self._request_completion(
                [
                    ChatMessage(
                        role=Role.SYSTEM,
                        content=self._build_personality_system_prompt(
                            "discord", reply_mode=resolved_reply_mode, retry_hint=retry_hint
                        ),
                    ),
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                        images=inline_images or None,
                    ),
                ]
            )
            return response
        except Exception as exc:
            logger.error(f"Error answering mention question: {exc}")
            return "I couldn't answer that right now."

    async def answer_social_question(
        self,
        *,
        platform: str,
        account_handle: str,
        question: str,
        messages: list[dict[str, Any]],
        current_message: dict[str, Any],
        max_chars: int,
        retry_hint: Optional[str] = None,
    ) -> str:
        has_reference = bool(current_message.get("reply_to_message_id"))
        reply_mode = self._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        user_payload = self._build_social_mention_payload(
            platform=platform,
            account_handle=account_handle,
            question=question,
            messages=messages,
            current_message=current_message,
            max_chars=max_chars,
            reply_mode=reply_mode,
            retry_hint=retry_hint,
        )
        try:
            response = self._request_completion(
                [
                    ChatMessage(
                        role=Role.SYSTEM,
                        content=self._build_personality_system_prompt(
                            platform, reply_mode=reply_mode, retry_hint=retry_hint
                        ),
                    ),
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                    ),
                ]
            )
            return response
        except Exception as exc:
            logger.error(f"Error answering social question: {exc}")
            return "I couldn't answer that right now."


class GenAIAnthropic(GenAIBase):
    client: Anthropic
    tokenizer: Tokenizer

    def __init__(self, settings: Settings) -> None:
        assert settings.genai.model.split("-")[0] in (
            "claude",
            "sonnet",
            "opus",
            "haiku",
        ), "GenAIAnthropic requires an Anthropic model"
        self.settings = settings
        self.client = Anthropic(api_key=settings.tokens.anthropic)
        self.tokenizer = Tokenizer.from_pretrained("Xenova/claude-tokenizer")

    def _request_completion(
        self, messages: list[ChatMessage], system_prompt: Optional[str] = None
    ) -> str:
        response = self.client.messages.create(
            model=self.settings.genai.model,
            max_tokens=self.settings.genai.tokens.output_max,
            system=system_prompt,
            messages=[
                {"role": message.role, "content": message.content} for message in messages
            ],  # type: ignore
        )
        logger.info(f"Anthropic response: {response.content}")
        if not response.content or not isinstance(response.content[0], TextBlock):
            logger.error(f"Unexpected response format: {response.content}")
            return "No response from Anthropic"
        return response.content[0].text or "No response from Anthropic"

    async def judge_debate(
        self,
        ctx: ApplicationContext | Reaction,
        participants: list[str],
        topic: Optional[str] = None,
    ) -> tuple[str, str]:
        messages_for_context = await self._collect_history_messages(
            ctx.message.channel if isinstance(ctx, Reaction) else ctx.channel,
            before=datetime.now(),
            after=datetime.now() - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
        )

        if not messages_for_context:
            logger.warning("No conversation history found.")
            return "error", "No conversation history available to judge."

        user_payload, inline_images = await self._build_debate_payload(
            ctx, messages_for_context, participants, topic=topic
        )
        try:
            response = self._request_completion(
                [
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                        images=inline_images or None,
                    )
                ],
                self._build_personality_system_prompt("discord"),
            )
            return self._parse_debate_response(response, participants)
        except Exception as exc:
            logger.error(f"Error occurred in judge_debate: {exc}")
            return "error", "An error occurred while judging the debate."

    async def answer_message_question(
        self,
        message: Message,
        question: str,
        user_ids: Optional[set[int]] = None,
        retry_hint: Optional[str] = None,
        history_before: Optional[datetime] = None,
        *,
        recent_context_human_turns: Optional[int] = None,
        merge_reply_chain: bool = True,
    ) -> str:
        has_reference = bool(message.reference and message.reference.message_id)
        reply_mode = self._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        _ctx_limit = (
            recent_context_human_turns
            if recent_context_human_turns is not None
            else self._mention_context_limit(
                question,
                settings.genai.question.recent_messages,
                has_reference=has_reference,
                retry_hint=retry_hint,
            )
        )
        messages_for_context = await self._collect_recent_context_messages(
            message,
            _ctx_limit,
            user_ids=user_ids,
            include_current=has_reference,
            history_before=history_before,
            merge_reply_chain=merge_reply_chain,
        )
        user_payload, inline_images, resolved_reply_mode = await self._build_mention_reply_payload(
            message,
            messages_for_context,
            question,
            user_ids=user_ids,
            reply_mode=reply_mode,
            retry_hint=retry_hint,
        )
        try:
            response = self._request_completion(
                [
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                        images=inline_images or None,
                    )
                ],
                self._build_personality_system_prompt(
                    "discord", reply_mode=resolved_reply_mode, retry_hint=retry_hint
                ),
            )
            return response
        except Exception as exc:
            logger.error(f"Error answering mention question: {exc}")
            return "I couldn't answer that right now."

    async def answer_social_question(
        self,
        *,
        platform: str,
        account_handle: str,
        question: str,
        messages: list[dict[str, Any]],
        current_message: dict[str, Any],
        max_chars: int,
        retry_hint: Optional[str] = None,
    ) -> str:
        has_reference = bool(current_message.get("reply_to_message_id"))
        reply_mode = self._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        user_payload = self._build_social_mention_payload(
            platform=platform,
            account_handle=account_handle,
            question=question,
            messages=messages,
            current_message=current_message,
            max_chars=max_chars,
            reply_mode=reply_mode,
            retry_hint=retry_hint,
        )
        try:
            response = self._request_completion(
                [
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                    )
                ],
                self._build_personality_system_prompt(
                    platform, reply_mode=reply_mode, retry_hint=retry_hint
                ),
            )
            return response
        except Exception as exc:
            logger.error(f"Error answering social question: {exc}")
            return "I couldn't answer that right now."


class _GenAILocalWithWebTools(GenAIBase):
    tokenizer: None
    _MAX_TOOL_ROUNDS = 4
    _MAX_TOOL_CALLS_PER_ROUND = 2
    _MAX_SOURCES_IN_REPLY = 3

    class _HTMLTextExtractor(HTMLParser):
        _SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside"}
        _PRIORITY_TAGS = {"main", "article"}
        _SKIP_ATTR_KEYWORDS = {
            "nav",
            "navigation",
            "sidebar",
            "footer",
            "header",
            "breadcrumb",
            "breadcrumbs",
            "toc",
            "table-of-contents",
            "menu",
            "related",
        }
        _BLOCK_TAGS = {
            "article",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "figcaption",
            "figure",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "hr",
            "li",
            "main",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "td",
            "th",
            "tr",
            "ul",
        }

        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self._skip_depth = 0
            self._skip_stack: list[str] = []
            self._priority_stack: list[str] = []
            self._priority_parts: list[str] = []
            self._fallback_parts: list[str] = []

        def _should_skip_attrs(
            self, tag: str, attrs: list[tuple[str, Optional[str]]]
        ) -> bool:
            if tag in self._PRIORITY_TAGS or not self._priority_stack:
                return False
            attr_map = {name.lower(): (value or "") for name, value in attrs}
            candidates = " ".join(
                [
                    attr_map.get("id", ""),
                    attr_map.get("class", ""),
                    attr_map.get("role", ""),
                    attr_map.get("aria-label", ""),
                ]
            ).lower()
            return any(keyword in candidates for keyword in self._SKIP_ATTR_KEYWORDS)

        def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
            tag = tag.lower()
            if tag in self._SKIP_TAGS or self._should_skip_attrs(tag, attrs):
                self._skip_depth += 1
                self._skip_stack.append(tag)
                return
            if self._skip_depth:
                return
            if tag in self._PRIORITY_TAGS:
                self._priority_stack.append(tag)
            if tag in self._BLOCK_TAGS:
                self._append_break()

        def handle_startendtag(
            self, tag: str, attrs: list[tuple[str, Optional[str]]]
        ) -> None:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

        def handle_endtag(self, tag: str) -> None:
            tag = tag.lower()
            if self._skip_depth:
                if self._skip_stack and tag == self._skip_stack[-1]:
                    self._skip_stack.pop()
                    self._skip_depth -= 1
                return
            if tag in self._BLOCK_TAGS:
                self._append_break()
            if tag in self._PRIORITY_TAGS and self._priority_stack:
                self._priority_stack.pop()

        def handle_data(self, data: str) -> None:
            if self._skip_depth:
                return
            text = unescape(data)
            if not text.strip():
                return
            target = self._priority_parts if self._priority_stack else self._fallback_parts
            target.append(text)

        def handle_entityref(self, name: str) -> None:
            self.handle_data(f"&{name};")

        def handle_charref(self, name: str) -> None:
            self.handle_data(f"&#{name};")

        def _append_break(self) -> None:
            target = self._priority_parts if self._priority_stack else self._fallback_parts
            if not target or target[-1] != "\n":
                target.append("\n")

        @staticmethod
        def _collapse(parts: list[str]) -> str:
            text = "".join(parts)
            text = text.replace("\r", "\n")
            text = re.sub(r"[ \t\f\v]+", " ", text)
            text = re.sub(r" *\n *", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

        def get_text(self) -> str:
            priority = self._collapse(self._priority_parts)
            if priority:
                return priority
            return self._collapse(self._fallback_parts)

    def _extract_answer_from_thinking(self, thinking: str) -> str:
        """When content is empty but thinking exists, try to extract a final answer."""
        text = thinking.strip()
        if not text:
            return ""
        # Prefer explicit "I'll say:" / "Option:" style endings
        for pattern in [
            r"[Ii]'ll say:\s*[\"']?([^\"'\n]+)[\"']?",
            r"[Oo]ption:\s*[\"']?([^\"'\n]+)[\"']?",
            r"[Ii]'ll go with:\s*[\"']?([^\"'\n]+)[\"']?",
        ]:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
        # Fallback: last quoted string (often the intended reply)
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return quoted[-1].strip()
        # Last non-empty line that looks like a reply (not meta-commentary)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        skip = {"i'll", "i'll say", "option:", "so", "therefore", "thus"}
        for ln in reversed(lines):
            if len(ln) > 10 and ln.lower().split()[0] not in skip:
                return ln
        return lines[-1] if lines else ""

    @staticmethod
    def _strip_think_block(content: str) -> str:
        """
        Remove leaked <think>...</think> blocks from visible assistant content.
        Some local templates include reasoning tags directly in content.
        """
        text = content.strip()
        if "<think>" in text and "</think>" in text:
            tail = text.split("</think>", 1)[1].strip()
            if tail:
                return tail
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
        return text

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = None
        self.tokenizer = None
        self._request_lock = asyncio.Lock()
        self._queued_local_requests = 0

    async def _run_local_request(self, source: str, runner: Any) -> str:
        loop = asyncio.get_running_loop()
        queued_at = loop.time()
        self._queued_local_requests += 1
        queued_behind = max(0, self._queued_local_requests - 1)
        if queued_behind:
            logger.info(
                f"local_request_queued source={source} queued_behind={queued_behind}"
            )
        try:
            async with self._request_lock:
                wait_ms = int((loop.time() - queued_at) * 1000)
                logger.info(
                    f"local_request_start source={source} "
                    f"queued_behind={queued_behind} wait_ms={wait_ms}"
                )
                started_at = loop.time()
                result = await runner()
                run_ms = int((loop.time() - started_at) * 1000)
                logger.info(f"local_request_done source={source} run_ms={run_ms}")
                return result
        finally:
            self._queued_local_requests = max(0, self._queued_local_requests - 1)

    async def _run_local_streaming(
        self,
        source: str,
        stream_factory: Callable[[], AsyncIterator[str]],
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queued_at = loop.time()
        self._queued_local_requests += 1
        queued_behind = max(0, self._queued_local_requests - 1)
        if queued_behind:
            logger.info(
                f"local_request_queued source={source} queued_behind={queued_behind}"
            )
        try:
            async with self._request_lock:
                wait_ms = int((loop.time() - queued_at) * 1000)
                logger.info(
                    f"local_request_start source={source} "
                    f"queued_behind={queued_behind} wait_ms={wait_ms}"
                )
                started_at = loop.time()
                stream = stream_factory()
                async for chunk in stream:
                    yield chunk
                run_ms = int((loop.time() - started_at) * 1000)
                logger.info(f"local_request_done source={source} run_ms={run_ms}")
        finally:
            self._queued_local_requests = max(0, self._queued_local_requests - 1)

    @staticmethod
    def _normalize_html_text(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value)
        value = unescape(value)
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _extract_duckduckgo_result_url(cls, href: str) -> str:
        href = unescape(href).strip()
        if href.startswith("//"):
            href = f"https:{href}"
        parsed = urlparse(href)
        redirect_target = parse_qs(parsed.query).get("uddg", [None])[0]
        if redirect_target:
            return redirect_target
        return href

    @classmethod
    def _parse_duckduckgo_lite_results(
        cls, html_text: str, max_results: int
    ) -> list[dict[str, str]]:
        """Parse https://lite.duckduckgo.com/lite/ HTML (result-link + result-snippet)."""
        results: list[dict[str, str]] = []
        for open_tag in re.finditer(r"<a\b([^>]*)>", html_text, flags=re.IGNORECASE):
            tag_attrs = open_tag.group(1)
            if not re.search(r"class\s*=\s*['\"]result-link['\"]", tag_attrs, re.IGNORECASE):
                continue
            href_m = re.search(r"\bhref\s*=\s*(['\"])([^'\"]*)\1", tag_attrs, re.IGNORECASE)
            if not href_m:
                continue
            href = href_m.group(2)
            start_content = open_tag.end()
            end_a = html_text.find("</a>", start_content)
            if end_a == -1:
                continue
            title = cls._normalize_html_text(html_text[start_content:end_a])
            url = cls._extract_duckduckgo_result_url(href)
            rest = html_text[end_a + len("</a>") :]
            snippet_m = re.search(
                r"<td[^>]*\bclass\s*=\s*['\"]result-snippet['\"][^>]*>(.*?)</td>",
                rest,
                flags=re.DOTALL | re.IGNORECASE,
            )
            snippet = (
                cls._normalize_html_text(snippet_m.group(1)) if snippet_m else ""
            )
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break
        return results

    @classmethod
    def _extract_html_title(cls, html_text: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.DOTALL | re.IGNORECASE)
        if match is None:
            return ""
        return cls._normalize_html_text(match.group(1))

    @classmethod
    def _html_to_text(cls, html_text: str) -> str:
        parser = cls._HTMLTextExtractor()
        parser.feed(html_text)
        parser.close()
        return parser.get_text()

    async def _fetch_url(self, url: str) -> str:
        timeout_seconds = self.settings.genai.web_search.timeout_seconds
        headers = {"User-Agent": "Mozilla/5.0"}
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return self._json_dumps({"url": url, "error": "fetch_url requires a valid http or https URL"})

        request = Request(url, headers=headers)

        def fetch_response() -> tuple[str, str]:
            with urlopen(request, timeout=timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read().decode(charset, "ignore")
                return content_type, body

        content_type, body = await asyncio.to_thread(fetch_response)
        title = ""
        excerpt = ""
        if "html" in content_type.lower():
            title = self._extract_html_title(body)
            excerpt = self._html_to_text(body)
        else:
            excerpt = re.sub(r"\s+", " ", body).strip()

        if len(excerpt) > 6000:
            excerpt = excerpt[:6000].rsplit(" ", 1)[0].rstrip() + "…"

        return self._json_dumps(
            {
                "url": url,
                "title": title,
                "content_type": content_type,
                "content": excerpt,
            }
        )

    def _build_tools(self) -> list[dict[str, Any]]:
        if not self.settings.genai.web_search.enabled:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": (
                        "Search the public web for current or external information and "
                        "return a few relevant results with snippets. Use for time-sensitive "
                        "or verifiable facts (news, stats, dates, current heads of state, etc.)."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The search query to look up on the web. Do not invent "
                                    "specific years or dates unless the user provided them. "
                                    "When a narrower source would help, use operators like "
                                    "site:github.com, site:wikipedia.org, or "
                                    "site:docs.python.org."
                                ),
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Optional number of results to return.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url",
                    "description": (
                        "Fetch a public URL and return readable page text so you can answer "
                        "questions about a user-provided link."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["url"],
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "A full http or https URL to fetch and read.",
                            }
                        },
                    },
                },
            },
        ]

    async def _search_web(
        self, query: str, requested_results: Optional[int] = None
    ) -> str:
        max_results = self.settings.genai.web_search.max_results
        if requested_results is not None:
            max_results = max(1, min(requested_results, max_results))

        timeout_seconds = self.settings.genai.web_search.timeout_seconds
        headers = {"User-Agent": "Mozilla/5.0"}

        async def run_search(search_query: str) -> list[dict[str, str]]:
            url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(search_query)}"
            request = Request(url, headers=headers)

            def fetch_html() -> str:
                with urlopen(request, timeout=timeout_seconds) as response:
                    return response.read().decode("utf-8", "ignore")

            html_text = await asyncio.to_thread(fetch_html)
            return self._parse_duckduckgo_lite_results(html_text, max_results)

        results = await run_search(query)
        simplified_query = re.sub(r"\b20\d{2}\b", " ", query)
        simplified_query = re.sub(r"\s+", " ", simplified_query).strip()
        if not results and simplified_query and simplified_query != query:
            results = await run_search(simplified_query)
        if not results:
            return self._json_dumps(
                {
                    "query": query,
                    "results": [],
                    "note": "No web results were found.",
                }
            )
        return self._json_dumps({"query": query, "results": results})

    async def _execute_parsed_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, str]:
        if name == "fetch_url":
            url = str(arguments.get("url") or "").strip()
            if not url:
                return name, self._json_dumps({"error": "fetch_url requires a url"})
            logger.info(f"LLM tool call: fetch_url url={url!r}")
            try:
                return name, await self._fetch_url(url)
            except Exception as exc:
                logger.warning(f"URL fetch failed for {url!r}: {exc}")
                return name, self._json_dumps({"url": url, "error": str(exc)})
        if name != "search_web":
            return name or "unknown_tool", self._json_dumps(
                {"error": f"Unsupported tool call: {name or 'unknown'}"}
            )

        query = str(arguments.get("query") or "").strip()
        if not query:
            return name, self._json_dumps({"error": "search_web requires a query"})

        requested_results: Optional[int] = None
        if arguments.get("max_results") is not None:
            try:
                requested_results = int(arguments["max_results"])
            except (TypeError, ValueError):
                requested_results = None

        logger.info(f"LLM tool call: search_web query={query!r}")
        try:
            return name, await self._search_web(query, requested_results)
        except Exception as exc:
            logger.warning(f"Web search failed for query {query!r}: {exc}")
            return name, self._json_dumps({"query": query, "error": str(exc)})

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> tuple[str, str]:
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "").strip()
        raw_args = function.get("arguments")
        if isinstance(raw_args, str):
            if raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    arguments = parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = {}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {}
        return await self._execute_parsed_tool(name, arguments)

    def _extract_sources_from_tool_result(
        self, tool_name: str, tool_result: str
    ) -> list[dict[str, str]]:
        parsed = self._parse_json_response(tool_result)
        if parsed is None:
            return []
        if tool_name == "fetch_url":
            url = str(parsed.get("url") or "").strip()
            title = str(parsed.get("title") or "").strip() or url
            if not url:
                return []
            return [{"title": title, "url": url}]
        if tool_name != "search_web":
            return []
        raw_results = parsed.get("results")
        if not isinstance(raw_results, list):
            return []

        sources: list[dict[str, str]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            sources.append({"title": title, "url": url})
            if len(sources) >= self._MAX_SOURCES_IN_REPLY:
                break
        return sources

    def _append_sources_to_response(
        self, content: str, sources: list[dict[str, str]]
    ) -> str:
        if not sources:
            return content

        source_lines = [
            f"{idx}. {source['title']} - <{source['url']}>"
            for idx, source in enumerate(sources, start=1)
        ]
        sources_block = "sources:\n" + "\n".join(source_lines)
        separator = "\n\n" if content.strip() else ""
        max_chars = 1999
        available = max_chars - len(separator) - len(sources_block)

        if available <= 0:
            trimmed_block = sources_block[:max_chars]
            return trimmed_block

        if len(content) > available:
            clipped = content[: max(0, available - 1)].rstrip()
            if clipped:
                content = clipped + "…"
            else:
                content = ""

        return f"{content}{separator}{sources_block}".strip()

    _PY_TOOL_CALL_RE = re.compile(
        r"\b(search_web|fetch_url)\(([^)]*)\)",
        flags=re.IGNORECASE,
    )
    _PY_TOOL_KV_RE = re.compile(
        r"""(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|([^,]+?))(?=\s*,|\s*$)""",
        flags=re.DOTALL,
    )

    @staticmethod
    def _strip_tool_call_blocks(content: str) -> str:
        cleaned = re.sub(
            r"<tool_call>\s*\{.*?\}\s*</tool_call>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = _GenAILocalWithWebTools._PY_TOOL_CALL_RE.sub("", cleaned)
        return cleaned.strip()

    def _parse_inline_tool_calls(self, content: str) -> list[dict[str, Any]]:
        parsed_calls: list[dict[str, Any]] = []
        blocks = re.findall(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        for idx, block in enumerate(blocks):
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            name = str(payload.get("name") or "").strip()
            if not name:
                continue
            raw_args = payload.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args_dict = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args_dict = {}
            elif isinstance(raw_args, dict):
                args_dict = raw_args
            else:
                args_dict = {}
            parsed_calls.append(
                {
                    "id": f"inline-tool-{idx}",
                    "name": name,
                    "arguments": args_dict if isinstance(args_dict, dict) else {},
                }
            )
            if len(parsed_calls) >= self._MAX_TOOL_CALLS_PER_ROUND:
                return parsed_calls

        # Python-style: search_web(query="...", max_results=3)
        for idx, m in enumerate(self._PY_TOOL_CALL_RE.finditer(content)):
            name = m.group(1).lower()
            args_blob = m.group(2)
            args_dict: dict[str, Any] = {}
            for kv in self._PY_TOOL_KV_RE.finditer(args_blob):
                key = kv.group(1)
                val = kv.group(2) if kv.group(2) is not None else (
                    kv.group(3) if kv.group(3) is not None else (kv.group(4) or "").strip()
                )
                if val == "":
                    continue
                if val.isdigit():
                    args_dict[key] = int(val)
                else:
                    args_dict[key] = val
            if not args_dict:
                continue
            parsed_calls.append(
                {
                    "id": f"inline-pycall-{idx}",
                    "name": name,
                    "arguments": args_dict,
                }
            )
            if len(parsed_calls) >= self._MAX_TOOL_CALLS_PER_ROUND:
                break
        return parsed_calls

    def _build_system_prompt(
        self,
        platform: str = "discord",
        *,
        tools_enabled: bool = False,
        reply_mode: Optional[str] = None,
        retry_hint: Optional[str] = None,
    ) -> str:
        prompt = self._build_personality_system_prompt(
            platform, reply_mode=reply_mode, retry_hint=retry_hint
        )
        if not tools_enabled:
            return prompt
        return (
            prompt
            + "\n\n[Tool Use]\n"
            + "You must call search_web when the user asks for current events, live facts, "
            + "official information, stats, dates, who currently holds office (for example "
            + "the president or head of state of a country), or other externally verifiable "
            + "facts that are not fully grounded in the chat transcript. Do not answer those "
            + "from memory alone. "
            + "When the current message, the replied-to message, or the immediate thread includes "
            + "a public URL and the user is asking about that link or its claims, call fetch_url "
            + "and read it before answering. "
            + "For casual opinions, jokes, or purely conversational replies that do not depend "
            + "on external facts, do not use search_web. "
            + "If a question is broad, technical, or could be answered from more than one angle, "
            + "you may call search_web twice in the same round with two meaningfully different "
            + "queries instead of waiting to fail first. Prefer complementary angles over "
            + "near-duplicate rewordings. "
            + "When building a search query, do not assume a year or date unless the user "
            + "explicitly gave one. "
            + "When you need official docs, factual references, or higher-signal technical "
            + "results, narrow the query with site: operators such as site:github.com, "
            + "site:wikipedia.org, or site:docs.python.org. "
            + "After using a tool, answer naturally in character and never mention tool names "
            + "or internal schemas."
        )

    async def judge_debate(
        self,
        ctx: ApplicationContext | Reaction,
        participants: list[str],
        topic: Optional[str] = None,
    ) -> tuple[str, str]:
        messages_for_context = await self._collect_history_messages(
            ctx.message.channel if isinstance(ctx, Reaction) else ctx.channel,
            before=datetime.now(),
            after=datetime.now() - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
        )

        if not messages_for_context:
            logger.warning("No conversation history found.")
            return "error", "No conversation history available to judge."

        user_payload, inline_images = await self._build_debate_payload(
            ctx, messages_for_context, participants, topic=topic
        )
        try:
            response = await self._request_completion(
                [
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                        images=inline_images or None,
                    )
                ],
                self._build_personality_system_prompt("discord"),
            )
            return self._parse_debate_response(response, participants)
        except Exception as exc:
            logger.error(f"Error occurred in judge_debate: {exc}")
            return "error", "An error occurred while judging the debate."

    async def answer_message_question(
        self,
        message: Message,
        question: str,
        user_ids: Optional[set[int]] = None,
        retry_hint: Optional[str] = None,
        history_before: Optional[datetime] = None,
        *,
        recent_context_human_turns: Optional[int] = None,
        merge_reply_chain: bool = True,
    ) -> str:
        has_reference = bool(message.reference and message.reference.message_id)
        reply_mode = self._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        has_image_attachments = self._VISION_ENABLED and self._message_has_images(message)
        if not has_image_attachments and has_reference and self._VISION_ENABLED:
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                has_image_attachments = self._message_has_images(ref)
            except Exception:
                pass
        if has_image_attachments:
            guild = message.guild
            bot_user_id: Optional[int] = guild.me.id if guild and guild.me else None
            chain = await self._collect_reference_chain_messages(
                message, bot_user_id, user_ids
            )
            if len(chain) > 3:
                chain = chain[-3:]
            messages_for_context = chain
            if has_reference:
                messages_for_context = list(messages_for_context)
                if message.id not in {m.id for m in messages_for_context}:
                    messages_for_context.append(message)
        else:
            _ctx_limit = (
                recent_context_human_turns
                if recent_context_human_turns is not None
                else self._mention_context_limit(
                    question,
                    settings.genai.question.recent_messages,
                    has_reference=has_reference,
                    retry_hint=retry_hint,
                )
            )
            messages_for_context = await self._collect_recent_context_messages(
                message,
                _ctx_limit,
                user_ids=user_ids,
                include_current=has_reference,
                history_before=history_before,
                merge_reply_chain=merge_reply_chain,
            )
        user_payload, inline_images, resolved_reply_mode = await self._build_mention_reply_payload(
            message,
            messages_for_context,
            question,
            user_ids=user_ids,
            reply_mode=reply_mode,
            retry_hint=retry_hint,
        )
        tools = self._build_tools()
        system_prompt = self._build_system_prompt(
            "discord", tools_enabled=bool(tools),
            reply_mode=resolved_reply_mode, retry_hint=retry_hint,
        )
        payload_tokens = self.count_tokens(user_payload)
        system_prompt_tokens = self.count_tokens(system_prompt)
        inline_image_chars = sum(len(img) for img in inline_images)
        logger.info(
            "discord_request_metrics "
            f"message={message.id} "
            f"reply_mode={reply_mode} "
            f"has_images={bool(inline_images)} "
            f"inline_images={len(inline_images)} "
            f"context_messages={len(messages_for_context)} "
            f"payload_tokens={payload_tokens} "
            f"system_prompt_tokens={system_prompt_tokens} "
            f"approx_prompt_tokens={payload_tokens + system_prompt_tokens} "
            f"inline_image_b64_chars={inline_image_chars} "
            f"question_chars={len(question)}"
        )
        try:
            response = await self._request_completion(
                [
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                        images=inline_images or None,
                    )
                ],
                system_prompt,
                tools=tools or None,
            )
            return response
        except Exception as exc:
            logger.error(f"Error answering mention question: {exc}")
            return "I couldn't answer that right now."

    async def answer_social_question(
        self,
        *,
        platform: str,
        account_handle: str,
        question: str,
        messages: list[dict[str, Any]],
        current_message: dict[str, Any],
        max_chars: int,
        retry_hint: Optional[str] = None,
    ) -> str:
        has_reference = bool(current_message.get("reply_to_message_id"))
        reply_mode = self._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        user_payload = self._build_social_mention_payload(
            platform=platform,
            account_handle=account_handle,
            question=question,
            messages=messages,
            current_message=current_message,
            max_chars=max_chars,
            reply_mode=reply_mode,
            retry_hint=retry_hint,
        )
        tools = self._build_tools()
        try:
            response = await self._request_completion(
                [
                    ChatMessage(
                        role=Role.USER,
                        content=user_payload,
                    )
                ],
                self._build_system_prompt(
                    platform, tools_enabled=bool(tools),
                    reply_mode=reply_mode, retry_hint=retry_hint,
                ),
                tools=tools or None,
            )
            return response
        except Exception as exc:
            logger.error(f"Error answering social question: {exc}")
            return "I couldn't answer that right now."


class GenAIOllama(_GenAILocalWithWebTools):
    async def _request_completion(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        async def run_request() -> str:
            payload_messages: list[dict[str, Any]] = []
            collected_sources: list[dict[str, str]] = []
            if system_prompt:
                payload_messages.append({"role": "system", "content": system_prompt.strip()})
            payload_messages.extend(message.to_dict() for message in messages)  # type: ignore[arg-type]

            url = f"{self.settings.genai.base_url.rstrip('/')}/api/chat"
            timeout = aiohttp.ClientTimeout(total=float(self.settings.genai.request_timeout))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for round_idx in range(self._MAX_TOOL_ROUNDS + 1):
                    num_predict = max(self.settings.genai.tokens.output_max, 2048)
                    payload = {
                        "model": self.settings.genai.model,
                        "messages": payload_messages,
                        "stream": False,
                        "options": {
                            "num_predict": num_predict,
                            "temperature": self.settings.genai.temperature,
                            "repeat_penalty": self.settings.genai.repeat_penalty,
                        },
                    }
                    if tools:
                        payload["tools"] = tools

                    async with session.post(url, json=payload) as response:
                        response.raise_for_status()
                        data = await response.json()

                    msg = data.get("message") or {}
                    tool_calls = msg.get("tool_calls") or []
                    if tools and tool_calls and round_idx < self._MAX_TOOL_ROUNDS:
                        assistant_message: dict[str, Any] = {"role": "assistant"}
                        if msg.get("content"):
                            assistant_message["content"] = msg["content"]
                        assistant_message["tool_calls"] = tool_calls
                        payload_messages.append(assistant_message)

                        for tool_call in tool_calls[: self._MAX_TOOL_CALLS_PER_ROUND]:
                            tool_name, tool_result = await self._execute_tool_call(tool_call)
                            for source in self._extract_sources_from_tool_result(
                                tool_name, tool_result
                            ):
                                if source not in collected_sources:
                                    collected_sources.append(source)
                            payload_messages.append(
                                {
                                    "role": "tool",
                                    "tool_name": tool_name,
                                    "content": tool_result,
                                }
                            )
                        continue

                    content = msg.get("content") or ""
                    if not content and msg.get("thinking"):
                        content = self._extract_answer_from_thinking(msg["thinking"])
                    if not content:
                        logger.error(f"Unexpected Ollama response: {data}")
                        return "not worth my time"
                    content = self._strip_think_block(content)
                    content = self._append_sources_to_response(content, collected_sources)
                    logger.info(f"Ollama response: {content}")
                    return content

            return "not worth my time"

        return await self._run_local_request("ollama_chat", run_request)


class GenAILlamaCpp(_GenAILocalWithWebTools):
    _VISION_ENABLED = True

    @staticmethod
    def _guess_mime(b64: str) -> str:
        try:
            header = base64.b64decode(b64[:32])
        except Exception:
            return "image/png"
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if header[:2] in (b"\xff\xd8",):
            return "image/jpeg"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
        if header[:4] == b"GIF8":
            return "image/gif"
        return "image/png"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        base_u = normalize_llamacpp_openai_base_url(settings.genai.base_url)
        self._openai_local = OpenAI(
            base_url=base_u,
            api_key="sk-no-key-required",
            timeout=float(settings.genai.request_timeout),
        )
        self.client = self._openai_local

    def _build_openai_messages(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
        *,
        include_images: bool = True,
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt.strip()})
        for message in messages:
            row = message.to_dict()
            role = row["role"].value if isinstance(row["role"], Role) else row["role"]
            images = (row.get("images") or []) if include_images else []
            if images:
                content_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": row["content"]}
                ]
                for b64 in images:
                    mime = self._guess_mime(b64)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                openai_messages.append({"role": role, "content": content_parts})
            else:
                openai_messages.append({"role": role, "content": row["content"]})
        return openai_messages

    async def _request_completion(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        async def run_request() -> str:
            has_images = any(m.images for m in messages)
            openai_messages = self._build_openai_messages(
                messages, system_prompt, include_images=has_images
            )

            collected_sources: list[dict[str, str]] = []
            temperature = float(self.settings.genai.temperature)
            repeat_penalty = float(self.settings.genai.repeat_penalty)
            extra_body: dict[str, Any] = {
                "repeat_penalty": repeat_penalty,
                "parse_tool_calls": True,
                "parallel_tool_calls": True,
            }

            max_tokens = max(self.settings.genai.tokens.output_max, 256)

            def _chat_create(msgs: list[dict[str, Any]], use_tools: bool):
                kwargs: dict[str, Any] = {
                    "model": self.settings.genai.model,
                    "messages": msgs,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extra_body": extra_body,
                }
                if use_tools and tools:
                    kwargs["tools"] = tools
                return self._openai_local.chat.completions.create(**kwargs)

            for round_idx in range(self._MAX_TOOL_ROUNDS + 1):
                try:
                    response = await asyncio.to_thread(
                        _chat_create, openai_messages, bool(tools)
                    )
                except Exception as exc:
                    if has_images and round_idx == 0 and "400" in str(exc):
                        logger.warning(
                            f"Image request failed ({exc}), retrying without images."
                        )
                        has_images = False
                        openai_messages = self._build_openai_messages(
                            messages, system_prompt, include_images=False
                        )
                        response = await asyncio.to_thread(
                            _chat_create, openai_messages, bool(tools)
                        )
                    else:
                        raise
                msg = response.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                parsed_calls: list[dict[str, Any]] = []
                for tc in tool_calls[: self._MAX_TOOL_CALLS_PER_ROUND]:
                    fn = tc.function
                    raw_args = fn.arguments or "{}"
                    try:
                        arguments = json.loads(raw_args) if raw_args.strip() else {}
                    except json.JSONDecodeError:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    parsed_calls.append(
                        {"id": tc.id, "name": fn.name, "arguments": arguments}
                    )

                inline_call_mode = False
                if tools and not parsed_calls and isinstance(msg.content, str):
                    parsed_calls = self._parse_inline_tool_calls(msg.content)
                    inline_call_mode = bool(parsed_calls)
                    if inline_call_mode:
                        logger.warning(
                            "llama.cpp returned inline <tool_call> content; using parser fallback."
                        )

                if tools and parsed_calls and round_idx < self._MAX_TOOL_ROUNDS:
                    assistant_message: dict[str, Any] = {"role": "assistant"}
                    if msg.content:
                        if inline_call_mode:
                            visible_content = self._strip_tool_call_blocks(msg.content)
                            if visible_content:
                                assistant_message["content"] = visible_content
                        else:
                            assistant_message["content"] = msg.content
                    serialized: list[dict[str, Any]] = []
                    for call in parsed_calls:
                        serialized.append(
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call["arguments"]),
                                },
                            }
                        )
                    assistant_message["tool_calls"] = serialized
                    openai_messages.append(assistant_message)

                    for call in parsed_calls:
                        arguments = call["arguments"]
                        tool_name, tool_result = await self._execute_parsed_tool(
                            call["name"], arguments
                        )
                        for source in self._extract_sources_from_tool_result(
                            tool_name, tool_result
                        ):
                            if source not in collected_sources:
                                collected_sources.append(source)
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": tool_result,
                            }
                        )
                    continue

                content = (msg.content or "").strip() if msg.content else ""
                if not content:
                    for attr in ("reasoning_content", "reasoning", "thinking"):
                        val = getattr(msg, attr, None)
                        if isinstance(val, str) and val.strip():
                            content = self._extract_answer_from_thinking(val)
                            break
                if not content and getattr(msg, "model_extra", None):
                    for key in ("reasoning_content", "reasoning", "thinking"):
                        val = (msg.model_extra or {}).get(key)
                        if isinstance(val, str) and val.strip():
                            content = self._extract_answer_from_thinking(val)
                            break
                if not content and tool_calls:
                    # Tool budget exhausted but model still wants to call tools.
                    # Force a final compose pass with no tools available.
                    logger.warning(
                        "Tool rounds exhausted with empty content; forcing "
                        "no-tool final compose pass."
                    )
                    final_response = await asyncio.to_thread(
                        _chat_create, openai_messages, False
                    )
                    final_msg = final_response.choices[0].message
                    content = (final_msg.content or "").strip() if final_msg.content else ""
                if not content:
                    logger.error(f"Unexpected llama.cpp chat response: {response!r}")
                    return "not worth my time"
                content = self._strip_think_block(content)
                content = self._append_sources_to_response(content, collected_sources)
                logger.info(f"llama.cpp response: {content}")
                return content

            return "not worth my time"

        return await self._run_local_request("llamacpp_chat", run_request)

    async def _openai_stream_deltas_async(
        self, openai_messages: list[dict[str, Any]]
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        exc_holder: list[BaseException] = []

        def worker() -> None:
            try:
                max_tokens = max(self.settings.genai.tokens.output_max, 256)
                stream = self._openai_local.chat.completions.create(
                    model=self.settings.genai.model,
                    messages=openai_messages,
                    temperature=float(self.settings.genai.temperature),
                    max_tokens=max_tokens,
                    stream=True,
                    extra_body={
                        "repeat_penalty": float(self.settings.genai.repeat_penalty),
                        "parse_tool_calls": False,
                        "parallel_tool_calls": False,
                    },
                )
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    ch0 = chunk.choices[0]
                    delta = getattr(ch0, "delta", None)
                    if delta is None:
                        continue
                    piece = getattr(delta, "content", None)
                    if piece:
                        fut = asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
                        fut.result(timeout=600)
            except BaseException as e:
                exc_holder.append(e)
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(
                        timeout=60
                    )
                except Exception:
                    pass

        exec_fut = loop.run_in_executor(None, worker)
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await exec_fut
        if exc_holder:
            raise exc_holder[0]

    async def answer_message_question_streaming(
        self,
        message: Message,
        question: str,
        user_ids: Optional[set[int]] = None,
        retry_hint: Optional[str] = None,
        history_before: Optional[datetime] = None,
        *,
        recent_context_human_turns: Optional[int] = None,
        merge_reply_chain: bool = True,
    ) -> AsyncIterator[str]:
        has_reference = bool(message.reference and message.reference.message_id)
        reply_mode = self._classify_mention_reply_mode(
            question, has_reference=has_reference, retry_hint=retry_hint
        )
        has_image_attachments = self._VISION_ENABLED and self._message_has_images(message)
        if not has_image_attachments and has_reference and self._VISION_ENABLED:
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                has_image_attachments = self._message_has_images(ref)
            except Exception:
                pass
        if has_image_attachments:
            guild = message.guild
            bot_user_id: Optional[int] = guild.me.id if guild and guild.me else None
            chain = await self._collect_reference_chain_messages(
                message, bot_user_id, user_ids
            )
            if len(chain) > 3:
                chain = chain[-3:]
            messages_for_context = chain
            if has_reference:
                messages_for_context = list(messages_for_context)
                if message.id not in {m.id for m in messages_for_context}:
                    messages_for_context.append(message)
        else:
            _ctx_limit_s = (
                recent_context_human_turns
                if recent_context_human_turns is not None
                else self._mention_context_limit(
                    question,
                    settings.genai.question.recent_messages,
                    has_reference=has_reference,
                    retry_hint=retry_hint,
                )
            )
            messages_for_context = await self._collect_recent_context_messages(
                message,
                _ctx_limit_s,
                user_ids=user_ids,
                include_current=has_reference,
                history_before=history_before,
                merge_reply_chain=merge_reply_chain,
            )
        user_payload, inline_images, resolved_reply_mode = await self._build_mention_reply_payload(
            message,
            messages_for_context,
            question,
            user_ids=user_ids,
            reply_mode=reply_mode,
            retry_hint=retry_hint,
            skip_images=True,
        )
        system_prompt = self._build_system_prompt(
            "discord",
            tools_enabled=False,
            reply_mode=resolved_reply_mode,
            retry_hint=retry_hint,
        )
        payload_tokens = self.count_tokens(user_payload)
        system_prompt_tokens = self.count_tokens(system_prompt)
        inline_image_chars = sum(len(img) for img in inline_images)
        logger.info(
            "discord_request_metrics "
            f"message={message.id} "
            f"reply_mode={reply_mode} "
            f"has_images={bool(inline_images)} "
            f"inline_images={len(inline_images)} "
            f"context_messages={len(messages_for_context)} "
            f"payload_tokens={payload_tokens} "
            f"system_prompt_tokens={system_prompt_tokens} "
            f"approx_prompt_tokens={payload_tokens + system_prompt_tokens} "
            f"inline_image_b64_chars={inline_image_chars} "
            f"question_chars={len(question)} "
            f"streaming=true"
        )
        openai_messages = self._build_openai_messages(
            [
                ChatMessage(
                    role=Role.USER,
                    content=user_payload,
                    images=None,
                )
            ],
            system_prompt,
            include_images=False,
        )

        async def body() -> AsyncIterator[str]:
            async for d in self._openai_stream_deltas_async(openai_messages):
                yield d

        def stream_factory() -> AsyncIterator[str]:
            return body()

        async for d in self._run_local_streaming(
            "llamacpp_chat_stream", stream_factory
        ):
            yield d


if settings.genai.model.startswith("llamacpp/"):
    settings.genai.model = settings.genai.model.split("/", 1)[1]  # type: ignore
    client = GenAILlamaCpp(settings)
elif settings.genai.model.startswith("ollama/"):
    settings.genai.model = settings.genai.model.split("/", 1)[1]  # type: ignore
    client = GenAIOllama(settings)
elif settings.genai.model.startswith("gpt"):
    client = GenAIGpt(settings)
elif settings.genai.model.split("-")[0] in (
    "claude",
    "sonnet",
    "opus",
    "haiku",
):
    client = GenAIAnthropic(settings)
else:
    raise ValueError("Unsupported model specified in settings")
