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
        reply_id = None
        if (
            message.reference
            and message.reference.resolved
            and isinstance(message.reference.resolved, Message)
        ):
            reply_id = message.reference.resolved.id
        id = message.id
        author = message.author.name
        if not reply_id:
            return f"[ID: {id} | {author}]: {message.content}"
        else:
            return f"[ID: {id} | {author} replying to {reply_id}]: {message.content}"

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
        pass


class GenAIGpt(GenAIBase):
    client: OpenAI
    tokenizer: Tokenizer

    def __init__(self, settings: Settings) -> None:
        assert settings.genai.model.startswith("gpt"), "GenAIGpt requires a GPT model"
        self.settings = settings
        self.client = OpenAI(api_key=settings.tokens.gpt)
        self.tokenizer = Tokenizer.from_pretrained("Xenova/gpt-4o")

    async def _build_prompt(
        self, conversation: str, system_prompt: str, command_prompt: str
    ) -> list[ChatMessage]:
        assert self.count_tokens(command_prompt) < 100
        messages = []
        if system_prompt:
            messages.append(
                ChatMessage(
                    role=Role.SYSTEM,
                    content=system_prompt.strip(),
                )
            )
        messages.append(
            ChatMessage(
                role=Role.USER,
                content=f"Based on the following conversation: \n\n{conversation}\n\n please answer: {command_prompt}.",
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
            response = self.client.chat.completions.create(
                model=settings.genai.model,
                messages=[message.to_dict() for message in messages],  # type: ignore
                max_tokens=settings.genai.tokens.output_max,
            )
            logger.info(f"GPT response: {response.choices[0].message.content}")
            return (
                response.choices[0].message.content
                if response.choices[0].message.content
                else "No response from GPT"
            )
        except Exception as e:
            logger.error(f"Error occurred in send_prompt: {e}")
            return "An error occurred while processing your request. Please try again later."


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

    async def _build_prompt(
        self, conversation: str, system_prompt: Optional[str], command_prompt: str
    ) -> list[ChatMessage]:
        assert self.count_tokens(command_prompt) < 100
        messages = []
        if system_prompt:
            logger.warning(
                "Internal error: System prompt passed to GenAIAnthropic build_prompt will be ignored."
            )
        messages.append(
            ChatMessage(
                role=Role.USER,
                content=f"Based on the following conversation: \n\n{conversation}\n\n please answer: {command_prompt}.",
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

        messages = await self._build_prompt(conversation, None, command_prompt)
        try:
            response = self.client.messages.create(
                model=self.settings.genai.model,
                max_tokens=self.settings.genai.tokens.output_max,
                system=system_prompt,
                messages=[message.to_dict() for message in messages],  # type: ignore
            )
            logger.info(f"Anthropic response: {response.content}")
            if not response.content and isinstance(response.content[0], TextBlock):
                logger.error(f"Unexpected response format: {response.content}")
            return (
                response.content[0].text  # type: ignore
                if response.content[0].text  # type: ignore
                else "No response from Anthropic"
            )
        except Exception as e:
            logger.error(f"Error occurred in send_prompt: {e}")
            if response:
                logger.error(f"Anthropic error response: {response}")
            return "An error occurred while processing your request. Please try again later."


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
