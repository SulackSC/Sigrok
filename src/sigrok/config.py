import os
from pathlib import Path
from typing import Any, Literal

import tomli
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings


class Tokens(BaseModel):
    bot: str
    gpt: str = ""
    hf: str = ""
    anthropic: str = ""


class BlueskySettings(BaseModel):
    enabled: bool = False
    # Login identifier, e.g. "alice.bsky.social"
    identifier: str = ""
    # App password (not your primary account password).
    password: str = ""
    api_base_url: str = "https://bsky.social"
    max_chars: int = 300
    poll_seconds: int = 30
    state_file: str = ".bluesky_state.json"
    thread_parent_height: int = 8


class XSettings(BaseModel):
    enabled: bool = False
    # OAuth2 "access token" (bearer) scoped for posting.
    bearer_token: str = ""
    api_base_url: str = "https://api.x.com"
    max_chars: int = 280


class SocialSettings(BaseModel):
    bluesky: BlueskySettings = Field(default_factory=BlueskySettings)
    x: XSettings = Field(default_factory=XSettings)


class DatabaseSettings(BaseModel):
    url: str
    echo: bool
    pool_size: int
    max_overflow: int
    pool_recycle: int
    pool_timeout: int
    backup_dir: str
    retention: int


class OwnerSettings(BaseModel):
    id: int


class IntentsSettings(BaseModel):
    guilds: bool
    messages: bool
    message_content: bool
    reactions: bool
    members: bool = False
    voice_states: bool = Field(
        default=False,
        validation_alias=AliasChoices("voice_states", "guild_voice_states"),
    )


class WhitelistEntry(BaseModel):
    guild: int
    channel: int
    roles: list[int]


class VoiceRecordSettings(BaseModel):
    """Chunked voice capture (see cogs.voice_rec)."""

    chunk_seconds: int = 30
    directory: str = "voice_recordings"
    announcement_file: str = ""
    announcement_interval_seconds: int = 60


class EventPostRule(BaseModel):
    """Post to a channel when a member joins or leaves (see cogs.conditional_posts)."""

    guild: int
    channel: int
    on: Literal["join", "leave"]
    message: str
    ignore_bots: bool = True


class TimedPostRule(BaseModel):
    """Post to a channel on a fixed interval (see cogs.conditional_posts)."""

    guild: int
    channel: int
    interval_minutes: int
    message: str


class BotSettings(BaseModel):
    prefix: str
    temp_dir: str
    cogs: list[str]
    owner: OwnerSettings
    intents: IntentsSettings
    whitelist: list[WhitelistEntry]
    voice_record: VoiceRecordSettings = Field(default_factory=VoiceRecordSettings)
    event_posts: list[EventPostRule] = Field(default_factory=list)
    timed_posts: list[TimedPostRule] = Field(default_factory=list)
    schedule_controller_user_ids: list[int] = Field(default_factory=list)


class GenaiHistorySettings(BaseModel):
    minutes: int
    messages: int


class GenaiQuestionSettings(BaseModel):
    recent_messages: int


class GenaiWebSearchSettings(BaseModel):
    enabled: bool = False
    max_results: int = 5
    timeout_seconds: int = 10


class GenaiDiscordStreamingSettings(BaseModel):
    enabled: bool = False
    edit_interval_seconds: float = 4.0


class GenaiTokenSettings(BaseModel):
    limit: int
    overhead_max: int
    output_max: int
    prompt_max: int


class GenaiSettings(BaseModel):
    model: str
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = Field(
        default=1.2,
        validation_alias=AliasChoices("temperature", "ollama_temperature"),
    )
    repeat_penalty: float = Field(
        default=1.2,
        validation_alias=AliasChoices("repeat_penalty", "ollama_repeat_penalty"),
    )
    request_timeout: float = 120.0
    tokens: GenaiTokenSettings
    history: GenaiHistorySettings
    question: GenaiQuestionSettings
    web_search: GenaiWebSearchSettings = GenaiWebSearchSettings()
    discord_streaming: GenaiDiscordStreamingSettings = Field(
        default_factory=GenaiDiscordStreamingSettings
    )


class Settings(BaseSettings):
    database: DatabaseSettings
    bot: BotSettings
    genai: GenaiSettings
    tokens: Tokens
    social: SocialSettings = Field(default_factory=SocialSettings)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomli.load(f)


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_settings() -> Settings:
    base_config = load_toml(Path("settings.toml"))
    if Path(".secrets.toml").exists():
        secrets_config = load_toml(Path(".secrets.toml"))
        merged = deep_merge(base_config, secrets_config)
    else:
        merged = base_config
    return Settings(**merged)


settings = load_settings()

if settings.tokens.hf:
    os.environ["HF_TOKEN"] = settings.tokens.hf


if __name__ == "__main__":
    print(settings.model_dump_json(indent=2))
