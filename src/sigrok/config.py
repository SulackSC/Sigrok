import os
from pathlib import Path
from typing import Any

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


class BotSettings(BaseModel):
    prefix: str
    temp_dir: str
    cogs: list[str]
    owner: OwnerSettings
    intents: IntentsSettings
    whitelist: list[WhitelistEntry]
    voice_record: VoiceRecordSettings = Field(default_factory=VoiceRecordSettings)


class GenaiHistorySettings(BaseModel):
    minutes: int
    messages: int


class GenaiQuestionSettings(BaseModel):
    recent_messages: int


class GenaiWebSearchSettings(BaseModel):
    enabled: bool = False
    max_results: int = 5
    timeout_seconds: int = 10


class GenaiRespectSettings(BaseModel):
    enabled: bool
    cooldown_seconds: int
    min_chars: int
    min_words: int
    max_delta_per_message: int


class GenaiTokenSettings(BaseModel):
    limit: int
    overhead_max: int
    output_max: int
    prompt_max: int


class GenaiSettings(BaseModel):
    model: str
    base_url: str = "http://127.0.0.1:11434"
    system_prompt: str
    temperature: float = Field(
        default=1.2,
        validation_alias=AliasChoices("temperature", "ollama_temperature"),
    )
    repeat_penalty: float = Field(
        default=1.2,
        validation_alias=AliasChoices("repeat_penalty", "ollama_repeat_penalty"),
    )
    tokens: GenaiTokenSettings
    history: GenaiHistorySettings
    question: GenaiQuestionSettings
    web_search: GenaiWebSearchSettings = GenaiWebSearchSettings()
    respect: GenaiRespectSettings


class EloSettings(BaseModel):
    scale: int
    max_delta: int


class Settings(BaseSettings):
    database: DatabaseSettings
    bot: BotSettings
    genai: GenaiSettings
    elo: EloSettings
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
