import aiogram
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.client.default import DefaultBotProperties
import dotenv
import os
import asyncio
from pathlib import Path
import logging

from model.bot.api import NerdLM
from model.preprocess.vocabulary_regist import Vocabulary

dotenv.load_dotenv(Path(__file__).resolve().parent.parent / "keys" / ".env")
token = os.getenv('TG_BOT_TOKEN')

dataset_path = Path(__file__).resolve().parents[2] / "datasets" / "english_qa" / "extended_qa_dataset.txt"
model = NerdLM(saved_model=True, saved_model_name='model/models/nerdlm.pt', inference=True)
vocab = Vocabulary()
VOCAB_PATH = Path(__file__).resolve().parents[2] / "datasets" / "vocabulary.json"
ENABLE_VOCAB_UPDATE = os.getenv("ENABLE_VOCAB_UPDATE", "0") == "1"

if not token:
    raise ValueError(
        "TG_BOT_TOKEN not found! Please ensure the .env file exists at '../../../model/api/keys/.env' and contains TG_BOT_TOKEN=your_token_here")

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
    if ENABLE_VOCAB_UPDATE:
        words = message.text.split()
        for word in words:
            vocab.add_word(word)
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        vocab.save_json_dump(str(VOCAB_PATH))

    output = model.generate_answer(message.text)

    await bot.send_message(message.chat.id, output)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Server started!")
    asyncio.run(main())
