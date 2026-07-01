from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from sigrok.cogs import backup as backup_module
from sigrok.db import ScheduleMentionJob, User, db_logger, ensure_user_schema, read_or_add_user


@pytest.mark.asyncio
async def test_db_logger_reraises_exceptions():
    @db_logger
    async def failing() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await failing()


@pytest.mark.asyncio
async def test_read_or_add_user_handles_integrity_error():
    with (
        patch("sigrok.db.ensure_user_schema", new=AsyncMock()),
        patch("sigrok.db.read_user", new=AsyncMock(side_effect=[None, User(guild_id=1, user_id=2)])),
        patch("sigrok.db.add_user", new=AsyncMock(side_effect=IntegrityError("insert", {}, Exception()))),
    ):
        user = await read_or_add_user(1, 2)
        assert user.guild_id == 1
        assert user.user_id == 2


def test_remove_old_backups_keeps_newest_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for idx in range(4):
            path = os.path.join(tmpdir, f"backup_2026010{idx}_120000.sqlite3.gz")
            with open(path, "wb") as handle:
                handle.write(b"x")
            os.utime(path, (idx + 1, idx + 1))
            paths.append(path)

        with patch.object(backup_module.settings.database, "backup_dir", tmpdir), patch.object(
            backup_module.settings.database, "retention", 2
        ):
            backup_module.remove_old_backups()

        remaining = sorted(os.listdir(tmpdir))
        assert len(remaining) == 2
        assert remaining[-1].startswith("backup_20260103_")


@pytest.mark.asyncio
async def test_read_due_schedule_jobs_filters_by_time():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    due_once = ScheduleMentionJob(
        job_id="once1",
        kind="once",
        guild_id=1,
        channel_id=2,
        message_id=3,
        creator_id=4,
        prompt="hi",
        due_at=now.replace(tzinfo=None) - timedelta(minutes=1),
        cron_expr=None,
        next_fire=None,
        ack_message_id=None,
    )
    future_once = ScheduleMentionJob(
        job_id="once2",
        kind="once",
        guild_id=1,
        channel_id=2,
        message_id=3,
        creator_id=4,
        prompt="later",
        due_at=now.replace(tzinfo=None) + timedelta(hours=1),
        cron_expr=None,
        next_fire=None,
        ack_message_id=None,
    )

    async def fake_execute(_stmt):
        class Result:
            def scalars(self):
                class ScalarResult:
                    def all(self_inner):
                        return [due_once]

                return ScalarResult()

        return Result()

    with (
        patch("sigrok.db.ensure_schedule_tables", new=AsyncMock()),
        patch("sigrok.db.get_session") as mock_session,
    ):
        session = AsyncMock()
        session.execute = fake_execute
        mock_session.return_value.__aenter__.return_value = session

        from sigrok import db

        jobs = await db.read_due_schedule_jobs(now)
        assert len(jobs) == 1
        assert jobs[0].job_id == "once1"
        assert future_once.job_id != jobs[0].job_id
