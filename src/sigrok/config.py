import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import tomli
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings


class YouTubeTokenSettings(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    refresh_token: str = ""


class KickTokenSettings(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    access_token: str = ""
    refresh_token: str = ""


class Tokens(BaseModel):
    bot: str
    gpt: str = ""
    hf: str = ""
    anthropic: str = ""
    opencode_go: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    twitch_bot_access_token: str = ""
    twitch_bot_refresh_token: str = ""
    youtube: YouTubeTokenSettings = Field(default_factory=YouTubeTokenSettings)
    kick: KickTokenSettings = Field(default_factory=KickTokenSettings)


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
    # OAuth2 user-context access token (tweet.read, tweet.write, users.read).
    bearer_token: str = ""
    api_base_url: str = "https://api.x.com"
    max_chars: int = 280
    poll_seconds: int = 30
    state_file: str = ".x_state.json"
    # Optional; resolved via GET /2/users/me on first use when empty.
    user_id: str = ""
    username: str = ""
    thread_parent_height: int = 8


class SocialSettings(BaseModel):
    bluesky: BlueskySettings = Field(default_factory=BlueskySettings)
    x: XSettings = Field(default_factory=XSettings)


class TwitchStreamingSettings(BaseModel):
    enabled: bool = False
    bot_username: str = "sigrok"
    bot_user_id: str = ""
    owner_user_id: str = ""
    channels: list[str] = Field(default_factory=list)
    max_chars: int = 500


class YouTubeStreamingSettings(BaseModel):
    enabled: bool = False
    bot_display_name: str = "Sigrok"
    channels: list[str] = Field(default_factory=list)
    max_chars: int = 200
    poll_fallback: bool = True


class KickStreamingSettings(BaseModel):
    enabled: bool = False
    bot_username: str = "sigrok"
    channels: list[str] = Field(default_factory=list)
    receive_mode: Literal["websocket", "webhook"] = "websocket"
    webhook_host: str = "127.0.0.1"
    webhook_public_key: str = ""
    webhook_port: int = 8765
    webhook_path: str = "/kick/eventsub"
    max_chars: int = 500


class StreamingSettings(BaseModel):
    enabled: bool = False
    global_reply_cooldown_seconds: float = 8.0
    per_user_cooldown_seconds: float = 60.0
    recent_messages_buffer: int = 40
    state_file: str = ".streaming_state.json"
    twitch: TwitchStreamingSettings = Field(default_factory=TwitchStreamingSettings)
    youtube: YouTubeStreamingSettings = Field(default_factory=YouTubeStreamingSettings)
    kick: KickStreamingSettings = Field(default_factory=KickStreamingSettings)


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


class SupersigrokSettings(BaseModel):
    """Paid SuperSigrok tier: DeepSeek V4 Flash Max Thinking instead of fast non-thinking.

    Grant via Discord user/role IDs (manual) or Stripe subscriptions (webhook → DB).
    """

    user_ids: list[int] = Field(default_factory=list)
    role_ids: list[int] = Field(default_factory=list)
    # Reasoning consumes completion budget; keep headroom above genai.tokens.output_max.
    output_max: int = 8192
    # Daily relationship reflection needs room for thinking + multi-user JSON.
    relationship_reflection_output_max: int = 32768
    # When set and still in the future, every Discord user gets Max Thinking.
    everyone_until: Optional[datetime] = None
    # Discord OAuth invite permissions integer (View Channel, Send, History, Embed,
    # Attach, Connect, Speak, View Audit Log, Add Reactions).
    invite_permissions: int = 3263680
    # Days to keep a subscriber guild after the sponsor's SuperSigrok lapses.
    guild_grace_days: int = 3
    # Global free-user @mention quota (SuperSigrok users skip).
    free_replies_per_window: int = 8
    free_window_minutes: int = 60
    # Max completion tokens for the in-character rate-limit cooldown reply.
    cheap_output_max: int = 80
    # Stripe Checkout (empty price_id → inline $5 USD/month price_data).
    stripe_price_id: str = ""
    stripe_amount_cents: int = 500
    stripe_currency: str = "usd"
    stripe_product_name: str = "SuperSigrok"
    stripe_success_url: str = "https://discord.com/channels/@me"
    stripe_cancel_url: str = "https://discord.com/channels/@me"
    stripe_webhook_host: str = "0.0.0.0"
    stripe_webhook_port: int = 8788
    stripe_webhook_path: str = "/stripe/webhook"


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
    supersigrok: SupersigrokSettings = Field(default_factory=SupersigrokSettings)


class GenaiHistorySettings(BaseModel):
    minutes: int
    messages: int


class GenaiQuestionSettings(BaseModel):
    recent_messages: int


class GenaiWebSearchSettings(BaseModel):
    enabled: bool = False
    max_results: int = 5
    timeout_seconds: int = 10
    provider: Literal["searxng"] = "searxng"
    base_url: str = "http://192.168.0.241:8080"
    language: str = "all"
    categories: str = "general"


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
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)


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
