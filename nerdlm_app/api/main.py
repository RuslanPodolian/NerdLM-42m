from email import message_from_binary_file

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.markdown import html_decoration as hd
import dotenv
import os
import asyncio
from pathlib import Path
import logging

from nerdlm_app.model.nerdlm import NerdLM
from nerdlm_app.model.vocabulary import Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_DIR = Path(__file__).resolve().parent

env_path = Path(os.getenv("BOT_ENV_PATH", str(API_DIR / "keys" / ".env")))
dotenv.load_dotenv(env_path)
token = os.getenv("TG_BOT_TOKEN")

model_path = Path(
    os.getenv("NERDLM_MODEL_PATH", str(PROJECT_ROOT  / 'nerdlm_app' / "nerdlm.pt"))
)
vocab_path = Path(
    os.getenv("NERDLM_VOCAB_PATH", str(PROJECT_ROOT / "nerdlm_app" / "model" / "datasets" / "vocabulary.json"))
)
enable_vocab_update = os.getenv("ENABLE_VOCAB_UPDATE", "0") == "1"

model = NerdLM(path='./nerdlm_app/model/datasets/english_extended_qa.txt', saved_model=True, saved_model_name=str(model_path), inference=True, training=False)
vocab = Vocabulary()

if not token:
    raise ValueError(
        "TG_BOT_TOKEN not found! Set it in the environment or add it to "
        f"'{env_path}' (or set BOT_ENV_PATH)."
    )

bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

dp = Dispatcher()

chat_requests = []

@dp.message(CommandStart())
async def start_command(message: Message):
    await bot.send_message(message.chat.id, f'Hello, {message.from_user.username}, it\'s NerdLM bot! Ask anything you want and get fast answer!')
    await bot.send_message(message.chat.id, f'Type /chat to start chatting with me!')

@dp.message(Command('chat'))
async def chat_command(message: Message):
    await bot.send_message(message.chat.id, f'Chat with me! Just type your question and I will answer it!')

@dp.message(F.text)
async def answer(message: Message):
    # In inference, avoid mutating/creating datasets unless explicitly enabled.
    if enable_vocab_update:
        words = message.text.split()
        for word in words:
            vocab.add_word(word)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save_json_dump(str(vocab_path))

    try:
        user_requests = []
        if len(chat_requests) == 0:
            for request in chat_requests:
                for user_message, user_id in request.items():
                    if user_id == message.from_user.id:
                        user_requests.append(user_message)

        output = model.generate_answer(message.text, previous_questions=user_requests, convert_indices_to_words=True)
    except Exception as e:
        output = "Something went wrong. Try another prompt."
        await bot.send_message(message.chat.id, output)
        raise e

    chat_requests.append({message.text: message.from_user.id})

    print(f"From user: {message.from_user.username} Bot send {message.text}")
    print(f"To user: {message.chat.username} Bot send {output}")
    print(f"Bot's answer size: {len(output)}")

    # Escape HTML entities to prevent parsing errors
    import html
    safe_output = html.escape(output)

    # Telegram message limit is 4096 characters
    if len(safe_output) > 4096:
        safe_output = safe_output[:4093] + "..."

    await bot.send_message(message.chat.id, safe_output)


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Server started!")
    asyncio.run(main())
