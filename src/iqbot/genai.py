import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from anthropic import Anthropic
from anthropic.types import TextBlock
from discord import ApplicationContext, Message, Reaction
from loguru import logger
from openai import OpenAI
from tokenizers import Tokenizer

from iqbot.config import Settings, settings


SIGROK_PERSONALITY_SYSTEM_PROMPT = """
You are Sigrok, a chronically online, mid-20s Discord veteran. Your overall vibe is dry, apathetic, and highly observant. You are not an AI; you are just another guy in the server. You don't try hard to be funny or edgy—your humor comes from blunt, cold pragmatism and pointing out the harsh realities of a situation. You rarely initiate topics; you prefer to react, judge, and analyze the "drama slop" of others.

[Worldview: "The Shortest Path"]

    Ruthless Pragmatism: You believe in total efficiency and logic ("the shortest path"). When people complain about emotional problems, relationship drama, or social issues, you strip away the feelings and look strictly at the raw, uncomfortable facts.

    The "Stats" Mindset: You view human interactions, especially dating and social hierarchies, almost like an RPG. You believe people are limited by their "stats" (height, income, age, hairline). You are naturally cynical about egalitarianism; you believe the dating market and social dynamics are inherently unequal and brutal.

    The Observer: You silently keep track of other users' weird habits, contradictions, and "lore." You categorize people mentally.

[Communication Style]

    Low Effort: Your typing style reflects boredom. Use mostly lowercase. Never use exclamation points unless being deeply sarcastic. Do not use periods at the end of your final sentence. Keep replies short—often just 1 to 3 sentences.

    Tone: Deadpan, mildly condescending, and detached. You are entirely devoid of customer-service politeness or AI helpfulness.

    Natural Slang: Sprinkle in terms like "L," "W," "smh," "rip," "slop," or "guy coded" only when it fits perfectly. Do not force slang into every sentence. Use "xD" or "lol" when someone fails or does something stupid.

    Address: Occasionally address the channel as "chat."

[Knowledge Base (Use ONLY if relevant)]

    You have passive, deep knowledge of internet streamer politics/drama (specifically DGG/Destiny).

    You know a lot about firearms and archery.

    You are a purist about cooking (specifically Italian sauces/lasagna) and finding cheap cuts of steak.

    You are familiar with heavy, grimdark fantasy literature (like R. Scott Bakker's Prince of Nothing series).

[Behavioral Directives]

    If someone asks for advice: Give the coldest, most unromantic, numbers-based answer possible.

    If someone is emotional: Be completely unfazed. Dismiss them or tell them to be more assertive.

    If someone is annoying: Give a one-word answer like "no" or "eh," or threaten to mute them.
"""


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage:
    role: Role
    content: str

    def __init__(self, role: Role, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return self.__dict__


class GenAIBase:
    client: OpenAI | Anthropic
    settings: Settings
    tokenizer: Tokenizer

    def __init__(self):
        raise NotImplementedError(
            "GenAIBase is an abstract class and cannot be instantiated directly."
        )

    def count_tokens(self, input: str) -> int:
        return len(self.tokenizer.encode(input))  # type: ignore

    def available_tokens(self, input: str) -> int:
        return (
            self.settings.genai.tokens.limit
            - self.settings.genai.tokens.overhead_max
            - self.settings.genai.tokens.prompt_max
            - self.settings.genai.tokens.output_max
            - self.count_tokens(input)
        )

    def format_message(self, message: Message) -> str:
        reply_id = message.reference.message_id if message.reference else None
        content = message.content.strip() or "[no text]"
        if message.attachments:
            attachment_names = ", ".join(attachment.filename for attachment in message.attachments)
            content = f"{content} [attachments: {attachment_names}]"
        if not reply_id:
            return f"[ID: {message.id} | {message.author.name}]: {content}"
        return (
            f"[ID: {message.id} | {message.author.name} replying to {reply_id}]: "
            f"{content}"
        )

    def _render_messages(self, messages: list[Message]) -> str:
        lines: list[str] = []
        remaining_tokens = self.available_tokens("")
        for message in reversed(messages):
            formatted_message = self.format_message(message)
            message_tokens = self.count_tokens(formatted_message)
            if remaining_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the message.")
                continue
            remaining_tokens -= message_tokens
            lines.append(formatted_message)
        return "\n".join(reversed(lines))

    async def _resolve_reply_message(self, message: Message) -> Optional[Message]:
        if not message.reference or message.reference.message_id is None:
            return None
        if message.reference.resolved and isinstance(message.reference.resolved, Message):
            return message.reference.resolved
        try:
            return await message.channel.fetch_message(message.reference.message_id)
        except Exception as exc:
            logger.debug(f"Unable to resolve reply message {message.reference.message_id}: {exc}")
            return None

    async def _collect_reply_chain(self, message: Message) -> list[Message]:
        chain: list[Message] = []
        seen_ids = {message.id}
        current = message
        while True:
            parent = await self._resolve_reply_message(current)
            if parent is None or parent.id in seen_ids:
                break
            seen_ids.add(parent.id)
            chain.append(parent)
            current = parent
        chain.reverse()
        return chain

    async def build_recent_context_for_message(
        self,
        source_message: Message,
        limit: int,
        user_ids: Optional[set[int]] = None,
        include_current: bool = False,
    ) -> str:
        anchor_messages: list[Message] = []
        if (
            include_current
            and not source_message.author.bot
        ):
            anchor_messages.append(source_message)

        history_limit = max(limit * 10, 50)
        async for message in source_message.channel.history(
            before=source_message.created_at,
            limit=history_limit,
            oldest_first=False,
        ):
            if message.author.bot:
                continue
            if user_ids and message.author.id not in user_ids:
                continue
            anchor_messages.append(message)
            if len(anchor_messages) >= limit:
                break

        if not anchor_messages:
            return ""

        anchor_messages.reverse()
        ordered_messages: dict[int, Message] = {}
        for anchor in anchor_messages:
            for chain_message in await self._collect_reply_chain(anchor):
                ordered_messages.setdefault(chain_message.id, chain_message)
            ordered_messages.setdefault(anchor.id, anchor)

        return self._render_messages(list(ordered_messages.values()))

    def _parse_respect_response(self, response: str) -> tuple[int, str]:
        delta_match = re.search(r"DELTA:\s*(-?1|0|1)\b", response, flags=re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", response, flags=re.IGNORECASE | re.DOTALL)
        if delta_match is None:
            return 0, "Model response did not contain a valid delta."
        delta = int(delta_match.group(1))
        delta = max(-self.settings.genai.respect.max_delta_per_message, delta)
        delta = min(self.settings.genai.respect.max_delta_per_message, delta)
        reason = reason_match.group(1).strip() if reason_match else "No reason provided."
        return delta, reason[:250]

    async def read_context(self, ctx: ApplicationContext | Reaction | Message) -> str:
        messages = []
        context_tokens = self.available_tokens("")
        if isinstance(ctx, ApplicationContext):
            channel = ctx.channel
        elif isinstance(ctx, Message):
            channel = ctx.channel
        elif isinstance(ctx, Reaction):
            channel = ctx.message.channel
        else:
            logger.error("Invalid context type provided.")
            return ""

        async for message in channel.history(
            before=datetime.now(),
            after=datetime.now() - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
            oldest_first=False,
        ):
            if message.author.bot:
                continue

            formatted_message = self.format_message(message)
            message_tokens = self.count_tokens(formatted_message)

            if context_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the message.")
                break

            context_tokens -= message_tokens
            messages.append(formatted_message)

        return "\n".join(messages[::-1])

    async def read_current_context(self, ctx: ApplicationContext) -> str:
        messages = []
        context_tokens = self.available_tokens("")
        async for message in ctx.channel.history(
            before=datetime.now(),
            after=datetime.now() - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
            oldest_first=False,
        ):
            if message.author.bot:
                continue

            formatted_message = self.format_message(message)
            message_tokens = self.count_tokens(formatted_message)

            if context_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the message.")
                break

            context_tokens -= message_tokens
            messages.append(formatted_message)

        return "\n".join(messages[::-1])

    async def read_message_context(self, msg: Message) -> str:
        messages = []
        context_tokens = self.available_tokens("")
        async for message in msg.channel.history(
            before=msg.created_at,
            after=msg.created_at - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
            oldest_first=False,
        ):
            if message.author.bot:
                continue

            formatted_message = self.format_message(message)
            message_tokens = self.count_tokens(formatted_message)

            if context_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the message.")
                break

            context_tokens -= message_tokens
            messages.append(formatted_message)

        return "\n".join(messages[::-1])

    async def read_reaction_context(self, reaction: Reaction) -> str:
        messages = []
        context_tokens = self.available_tokens("")
        async for message in reaction.message.channel.history(
            before=reaction.message.created_at,
            after=reaction.message.created_at
            - timedelta(minutes=settings.genai.history.minutes),
            limit=settings.genai.history.messages,
            oldest_first=False,
        ):
            if message.author.bot:
                continue

            formatted_message = self.format_message(message)
            message_tokens = self.count_tokens(formatted_message)

            if context_tokens - message_tokens < 0:
                logger.warning("Not enough tokens available for the message.")
                break

            context_tokens -= message_tokens
            messages.append(formatted_message)

        return "\n".join(messages[::-1])

    async def send_prompt(
        self,
        ctx: ApplicationContext | Reaction,
        system_prompt: str,
        command_prompt: str,
    ) -> str:
        raise NotImplementedError("send_prompt must be implemented by subclasses")

    async def answer_message_question(
        self, message: Message, question: str, user_ids: Optional[set[int]] = None
    ) -> str:
        raise NotImplementedError("answer_message_question must be implemented by subclasses")

    async def score_message_respect(self, message: Message) -> tuple[int, str]:
        raise NotImplementedError("score_message_respect must be implemented by subclasses")


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
            messages=[message.to_dict() for message in messages],  # type: ignore
            max_tokens=settings.genai.tokens.output_max,
        )
        content = response.choices[0].message.content
        logger.info(f"GPT response: {content}")
        return content if content else "No response from GPT"

    async def _build_prompt(
        self, conversation: str, system_prompt: str, command_prompt: str
    ) -> list[ChatMessage]:
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role=Role.SYSTEM, content=system_prompt.strip()))
        messages.append(
            ChatMessage(
                role=Role.USER,
                content=(
                    f"Based on the following conversation: \n\n{conversation}\n\n"
                    f" please answer: {command_prompt}."
                ),
            )
        )
        return messages

    async def send_prompt(
        self,
        ctx: ApplicationContext | Reaction,
        system_prompt: str,
        command_prompt: str,
    ) -> str:
        conversation = await self.read_context(ctx)

        if not conversation:
            logger.warning("No conversation history found.")
            return "No conversation history available to generate a response."

        messages = await self._build_prompt(conversation, system_prompt, command_prompt)
        try:
            return self._request_completion(messages)
        except Exception as exc:
            logger.error(f"Error occurred in send_prompt: {exc}")
            return "An error occurred while processing your request. Please try again later."

    async def answer_message_question(
        self, message: Message, question: str, user_ids: Optional[set[int]] = None
    ) -> str:
        conversation = await self.build_recent_context_for_message(
            message,
            settings.genai.question.recent_messages,
            user_ids=user_ids,
            include_current=bool(message.reference and message.reference.message_id),
        )
        system_prompt = (
            SIGROK_PERSONALITY_SYSTEM_PROMPT.strip()
            + "\n\n[Sigrok Q&A task]\n"
            + "Answer the user's question.\n"
            + "When the question refers to what was said in the recent conversation, use ONLY what appears in the supplied transcript.\n"
            + "If the question is unrelated (ex: 'what is a pineapple?'), answer using general knowledge normally.\n"
            + "If the question asks about specific people but they do not appear in the supplied context, say so briefly.\n"
            + "Hard constraint: the entire response must be <= 2000 characters.\n"
        )
        user_prompt = (
            f"Recent Discord conversation:\n{conversation or '[No recent conversation found.]'}\n\n"
            f"Question: {question}"
        )
        try:
            return self._request_completion(
                [
                    ChatMessage(role=Role.SYSTEM, content=system_prompt),
                    ChatMessage(role=Role.USER, content=user_prompt),
                ]
            )
        except Exception as exc:
            logger.error(f"Error answering mention question: {exc}")
            return "I couldn't answer that right now."

    async def score_message_respect(self, message: Message) -> tuple[int, str]:
        conversation = await self.build_recent_context_for_message(
            message,
            settings.genai.question.recent_messages,
            include_current=True,
        )
        system_prompt = (
            SIGROK_PERSONALITY_SYSTEM_PROMPT.strip()
            + "\n\n[Respect scoring task]\n"
            + "Evaluate whether the author's latest Discord message should change their IQ/respect score.\n"
            + "Be conservative: most messages should yield DELTA: 0.\n"
            + "Reward clear, substantive, insightful, well-reasoned, or genuinely informative messages.\n"
            + "Penalize clearly dishonest, incoherent, lazy, or aggressively low-quality messages.\n"
            + "Ignore ordinary banter, short acknowledgements, memes, and messages without enough substance.\n"
            + "Output EXACTLY two lines and nothing else:\n"
            + "DELTA: -1 or 0 or 1\n"
            + "REASON: <brief reason>\n"
        )
        user_prompt = (
            f"Conversation context:\n{conversation or self.format_message(message)}\n\n"
            f"Target author: {message.author.name}\n"
            f"Latest message ID: {message.id}"
        )
        try:
            response = self._request_completion(
                [
                    ChatMessage(role=Role.SYSTEM, content=system_prompt),
                    ChatMessage(role=Role.USER, content=user_prompt),
                ]
            )
            return self._parse_respect_response(response)
        except Exception as exc:
            logger.error(f"Error scoring respect: {exc}")
            return 0, "Respect scoring failed."


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
            messages=[message.to_dict() for message in messages],  # type: ignore
        )
        logger.info(f"Anthropic response: {response.content}")
        if not response.content or not isinstance(response.content[0], TextBlock):
            logger.error(f"Unexpected response format: {response.content}")
            return "No response from Anthropic"
        return response.content[0].text or "No response from Anthropic"

    async def _build_prompt(
        self, conversation: str, system_prompt: Optional[str], command_prompt: str
    ) -> list[ChatMessage]:
        if system_prompt:
            logger.warning(
                "Internal error: System prompt passed to GenAIAnthropic build_prompt will be ignored."
            )
        return [
            ChatMessage(
                role=Role.USER,
                content=(
                    f"Based on the following conversation: \n\n{conversation}\n\n"
                    f" please answer: {command_prompt}."
                ),
            )
        ]

    async def send_prompt(
        self,
        ctx: ApplicationContext | Reaction,
        system_prompt: str,
        command_prompt: str,
    ) -> str:
        conversation = await self.read_context(ctx)

        if not conversation:
            logger.warning("No conversation history found.")
            return "No conversation history available to generate a response."

        messages = await self._build_prompt(conversation, None, command_prompt)
        try:
            return self._request_completion(messages, system_prompt)
        except Exception as exc:
            logger.error(f"Error occurred in send_prompt: {exc}")
            return "An error occurred while processing your request. Please try again later."

    async def answer_message_question(
        self, message: Message, question: str, user_ids: Optional[set[int]] = None
    ) -> str:
        conversation = await self.build_recent_context_for_message(
            message,
            settings.genai.question.recent_messages,
            user_ids=user_ids,
            include_current=bool(message.reference and message.reference.message_id),
        )
        system_prompt = (
            SIGROK_PERSONALITY_SYSTEM_PROMPT.strip()
            + "\n\n[Sigrok Q&A task]\n"
            + "Answer the user's question.\n"
            + "When the question refers to what was said in the recent conversation, use ONLY what appears in the supplied transcript.\n"
            + "If the question is unrelated (ex: 'what is a pineapple?'), answer using general knowledge normally.\n"
            + "If the question asks about specific people but they do not appear in the supplied context, say so briefly.\n"
            + "Hard constraint: the entire response must be <= 2000 characters.\n"
        )
        user_prompt = (
            f"Recent Discord conversation:\n{conversation or '[No recent conversation found.]'}\n\n"
            f"Question: {question}"
        )
        try:
            return self._request_completion(
                [ChatMessage(role=Role.USER, content=user_prompt)],
                system_prompt,
            )
        except Exception as exc:
            logger.error(f"Error answering mention question: {exc}")
            return "I couldn't answer that right now."

    async def score_message_respect(self, message: Message) -> tuple[int, str]:
        conversation = await self.build_recent_context_for_message(
            message,
            settings.genai.question.recent_messages,
            include_current=True,
        )
        system_prompt = (
            SIGROK_PERSONALITY_SYSTEM_PROMPT.strip()
            + "\n\n[Respect scoring task]\n"
            + "Evaluate whether the author's latest Discord message should change their IQ/respect score.\n"
            + "Be conservative: most messages should yield DELTA: 0.\n"
            + "Reward clear, substantive, insightful, well-reasoned, or genuinely informative messages.\n"
            + "Penalize clearly dishonest, incoherent, lazy, or aggressively low-quality messages.\n"
            + "Ignore ordinary banter, short acknowledgements, memes, and messages without enough substance.\n"
            + "Output EXACTLY two lines and nothing else:\n"
            + "DELTA: -1 or 0 or 1\n"
            + "REASON: <brief reason>\n"
        )
        user_prompt = (
            f"Conversation context:\n{conversation or self.format_message(message)}\n\n"
            f"Target author: {message.author.name}\n"
            f"Latest message ID: {message.id}"
        )
        try:
            response = self._request_completion(
                [ChatMessage(role=Role.USER, content=user_prompt)],
                system_prompt,
            )
            return self._parse_respect_response(response)
        except Exception as exc:
            logger.error(f"Error scoring respect: {exc}")
            return 0, "Respect scoring failed."


if settings.genai.model.startswith("gpt"):
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
