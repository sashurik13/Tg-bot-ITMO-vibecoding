import argparse
import asyncio
import logging

from aiogram import Dispatcher

from app.agent.runtime import agent_runtime
from app.config import ConfigError, Settings
from app.db import create_pool
from app.handlers.reminders import router as reminders_router
from app.handlers.study import router
from app.health import HealthState, start_health_server
from app.llm import OpenAILLMClient
from app.logging_setup import configure_logging
from app.telegram import create_bot

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    llm = None
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        llm = OpenAILLMClient(settings)

        dispatcher = Dispatcher()
        dispatcher.include_router(reminders_router)
        dispatcher.include_router(router)

        runner = await start_health_server(state, settings.health_port)
        async with agent_runtime(settings, llm, state.pool) as agent:
            state.polling_task = asyncio.create_task(
                dispatcher.start_polling(
                    bot,
                    db=state.pool,
                    llm=llm,
                    agent=agent,
                    config=settings,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                    close_bot_session=False,
                )
            )
            state.initialized = True
            try:
                await state.polling_task
            finally:
                if not state.polling_task.done():
                    state.polling_task.cancel()
                    await asyncio.gather(state.polling_task, return_exceptions=True)
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await bot.session.close()
        if llm:
            await llm.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Учебный Telegram AI-ассистент")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен и прокси.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
