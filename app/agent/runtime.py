"""Own the stdio context on one task, with safe startup degradation."""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from app.agent.clock import SystemClock
from app.agent.mcp_client import connect_stdio, safe_error
from app.agent.mcp_launch import trusted_stdio_launch
from app.agent.reminders import ReminderService
from app.agent.repository import AgentRepository
from app.agent.service import ReadAgent
from app.db import create_pool

logger = logging.getLogger("app.agent.audit")


@asynccontextmanager
async def agent_runtime(settings, llm, pool):
    if not settings.lab2.enabled:
        yield None
        return
    clock = SystemClock()
    async with AsyncExitStack() as stack:
        coordination = await create_pool(settings)

        async def close_coordination():
            try:
                async with asyncio.timeout(10):
                    await coordination.close()
            except TimeoutError:
                coordination.terminate()

        stack.push_async_callback(close_coordination)
        repository = AgentRepository(
            pool,
            clock,
            settings.lab2.confirmation_ttl,
            coordination_pool=coordination,
        )
        launch, authority = trusted_stdio_launch(settings, clock)
        catalog = None
        try:
            catalog = await stack.enter_async_context(connect_stdio(launch, clock))
        except Exception as exc:
            logger.warning(
                "mcp_startup time=%s status=unavailable reason=%s",
                clock.now().isoformat(),
                safe_error(exc).code.value,
            )
        yield ReadAgent(
            llm,
            catalog,
            repository,
            clock,
            limit=settings.lab2.max_mcp_calls,
            schedule_timezone=settings.lab2.schedule_timezone,
            preparation=ReminderService(repository, catalog, authority, clock),
        )
