import asyncio
import logging
import os
import dotenv
import sys

from aiogram import Bot, Dispatcher
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.chat_action import ChatActionMiddleware 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from bot.handlers import router


# loading the token as an environment variable
dotenv.load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")


async def main():
    # parse_mode, it is responsible for the default message markup
    bot = Bot(token = BOT_TOKEN, parse_mode = ParseMode.HTML)
    # the bot data that we do not save in the database will be erased when restarting
    dp = Dispatcher(storage = MemoryStorage())
    dp.message.middleware(ChatActionMiddleware())
    dp.include_router(router)
    # deletes all updates that occurred after the last shutdown of the bot
    await bot.delete_webhook(drop_pending_updates = True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    asyncio.run(main())