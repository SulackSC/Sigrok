import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from pprint import pformat
from typing import AsyncIterator, Optional

from loguru import logger
from sqlalchemy import String, Text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from sigrok.config import settings

Base = declarative_base()

engine = create_async_engine(
    settings.database.url,
    echo=settings.database.echo,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_recycle=settings.database.pool_recycle,
    pool_timeout=settings.database.pool_timeout,
)

async_session = async_sessionmaker(engine, expire_on_commit=False)


class User(Base):
    __tablename__ = "users"
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(index=True)
    user_id: Mapped[int] = mapped_column(index=True)
    rating: Mapped[Optional[int]] = mapped_column(default=100)
    is_present: Mapped[bool] = mapped_column(default=True)

    def __repr__(self):
        return pformat(self.to_dict())

    def to_dict(self) -> dict:
        return self.__dict__


class ScheduleMentionJob(Base):
    """Persisted @Sigrok @schedule mention jobs (one-shot and cron)."""

    __tablename__ = "schedule_mention_jobs"
    __allow_unmapped__ = True

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(8), index=True)  # "once" | "cron"
    guild_id: Mapped[int] = mapped_column(index=True)
    channel_id: Mapped[int] = mapped_column(index=True)
    message_id: Mapped[int] = mapped_column(index=True)
    creator_id: Mapped[int] = mapped_column(index=True)
    prompt: Mapped[str] = mapped_column(Text())
    due_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    cron_expr: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    next_fire: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    ack_message_id: Mapped[Optional[int]] = mapped_column(nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<ScheduleMentionJob {self.job_id} {self.kind}>"


_schedule_tables_created = False


def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


async def ensure_schedule_tables() -> None:
    """Create schedule_mention_jobs if missing (does not drop existing tables)."""
    global _schedule_tables_created
    if _schedule_tables_created:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _schedule_tables_created = True


async def insert_schedule_mention_job(
    *,
    job_id: str,
    kind: str,
    guild_id: int,
    channel_id: int,
    message_id: int,
    creator_id: int,
    prompt: str,
    due_at: Optional[datetime],
    cron_expr: Optional[str],
    next_fire: Optional[datetime],
    ack_message_id: Optional[int],
) -> None:
    await ensure_schedule_tables()
    row = ScheduleMentionJob(
        job_id=job_id,
        kind=kind,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        creator_id=creator_id,
        prompt=prompt,
        due_at=_to_naive_utc(due_at),
        cron_expr=cron_expr,
        next_fire=_to_naive_utc(next_fire),
        ack_message_id=ack_message_id,
    )
    async with get_session() as session:
        session.add(row)
        await session.commit()


async def delete_schedule_mention_job(job_id: str) -> None:
    await ensure_schedule_tables()
    async with get_session() as session:
        result = await session.execute(
            select(ScheduleMentionJob).where(ScheduleMentionJob.job_id == job_id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await session.delete(row)
            await session.commit()


async def list_schedule_mention_jobs() -> list[ScheduleMentionJob]:
    await ensure_schedule_tables()
    async with get_session() as session:
        result = await session.execute(select(ScheduleMentionJob))
        return list(result.scalars().all())


async def update_schedule_cron_next_fire(job_id: str, next_fire: datetime) -> None:
    await ensure_schedule_tables()
    async with get_session() as session:
        result = await session.execute(
            select(ScheduleMentionJob).where(ScheduleMentionJob.job_id == job_id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            row.next_fire = _to_naive_utc(next_fire)
            await session.commit()


def db_logger(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None

    return wrapper


@asynccontextmanager
async def get_session():
    async with async_session() as session:
        # async with session.begin():
        yield session


@db_logger
async def add_user(user: User) -> None:
    async with get_session() as session:
        session.add(user)
        await session.commit()


@db_logger
async def read_user(guild_id: int, user_id: int) -> Optional[User]:
    async with get_session() as session:
        result = await session.execute(
            select(User).where(User.guild_id == guild_id, User.user_id == user_id)
        )
        return result.scalar_one_or_none()


@db_logger
async def read_or_add_user(guild_id: int, user_id: int) -> User:
    user = await read_user(guild_id, user_id)
    if user is None:
        user = User(guild_id=guild_id, user_id=user_id, rating=100)
        await add_user(user)
        logger.info(f"New user created: {user}")
    else:
        logger.info(f"Existing user found: {user}")
    return user


@db_logger
async def read_or_add_users(guild_id: int, user_ids: list[int]) -> list[User]:
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.user_id.in_(user_ids))
        result = await session.execute(stmt)
        existing_users = {user.user_id: user for user in result.scalars().all()}

        missing_user_ids = set(user_ids) - existing_users.keys()
        new_users = [
            User(guild_id=guild_id, user_id=user_id, rating=100)
            for user_id in missing_user_ids
        ]

        session.add_all(new_users)
        await session.commit()
        return list(existing_users.values()) + new_users


@db_logger
async def upsert_user_rating(guild_id: int, user_id: int, rating: int) -> User:
    async with get_session() as session:
        stmt = select(User).where(User.user_id == user_id, User.guild_id == guild_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            user.rating = rating
        else:
            user = User(user_id=user_id, guild_id=guild_id, rating=rating)
            session.add(user)

        await session.commit()
    return user


@db_logger
async def adjust_user_rating(guild_id: int, user_id: int, delta: int) -> User:
    async with get_session() as session:
        stmt = select(User).where(User.user_id == user_id, User.guild_id == guild_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            user = User(user_id=user_id, guild_id=guild_id, rating=max(100 + delta, 0))
            session.add(user)
        else:
            base = user.rating if user.rating is not None else 100
            user.rating = max(base + delta, 0)

        await session.commit()
    return user


async def read_top_ratings(guild_id: int) -> AsyncIterator[User]:
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.is_present)
        stmt = stmt.order_by(User.rating.desc(), User.user_id.asc())
        result = await session.stream(stmt)
        async for user in result.scalars():
            yield user


async def read_bottom_ratings(guild_id: int) -> AsyncIterator[User]:
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.is_present)
        stmt = stmt.order_by(User.rating.asc(), User.user_id.asc())
        result = await session.stream(stmt)
        async for user in result.scalars():
            yield user


@db_logger
async def read_present_users(guild_id: int) -> list[User]:
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.is_present)
        stmt = stmt.order_by(User.rating.desc(), User.user_id.asc())
        result = await session.execute(stmt)
        return list(result.scalars().all())


@db_logger
async def remove_user(guild_id: int, user_id: int) -> None:
    async with get_session() as session:
        stmt = select(User).where(User.user_id == user_id, User.guild_id == guild_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            await session.delete(user)
            await session.commit()
            logger.info(f"User {user} removed from the database")
        else:
            logger.warning(f"User {user_id} not found in the database")


async def async_main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(async_main())
