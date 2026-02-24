from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.client.default import DefaultBotProperties
import dotenv
import os
import asyncio
from pathlib import Path
import logging

from app.model.nerdlm import NerdLM
from app.model.vocabulary import Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_DIR = Path(__file__).resolve().parent

env_path = Path(os.getenv("BOT_ENV_PATH", str(API_DIR / "keys" / ".env")))
dotenv.load_dotenv(env_path)
token = os.getenv("TG_BOT_TOKEN")

model_path = Path(
    os.getenv("NERDLM_MODEL_PATH", str(PROJECT_ROOT / "app" / "model" / "saved_models" / "nerdlm.pt"))
)
vocab_path = Path(
    os.getenv("NERDLM_VOCAB_PATH", str(PROJECT_ROOT / "app" / "model" / "datasets" / "vocabulary.json"))
)
enable_vocab_update = os.getenv("ENABLE_VOCAB_UPDATE", "0") == "1"

model = NerdLM(saved_model=True, saved_model_name=str(model_path), inference=True, training=False)
vocab = Vocabulary()

if not token:
    raise ValueError(
        "TG_BOT_TOKEN not found! Set it in the environment or add it to "
        f"'{env_path}' (or set BOT_ENV_PATH)."
    )

bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

dp = Dispatcher()

@dp.message(CommandStart())
async def start_command(message: Message):
    await bot.send_message(message.chat.id, f'Hello, {message.from_user.username}, it\'s NerdLM bot! Ask anything you want and get fast answer!')
    await bot.send_message(message.chat.id, f'Type /chat to start chatting with me!')

@dp.message(Command('chat'))
async def chat_command(message: Message):
    await bot.send_message(message.chat.id, f'Chat with me! Just type your question and I will answer it!')

@dp.message()
async def answer(message: Message):
    # In inference, avoid mutating/creating datasets unless explicitly enabled.
    if enable_vocab_update:
        words = message.text.split()
        for word in words:
            vocab.add_word(word)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save_json_dump(str(vocab_path))

    output = model.generate_answer(message.text)

    await bot.send_message(message.chat.id, output)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Server started!")
    asyncio.run(main())
