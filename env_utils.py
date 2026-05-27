import os

from dotenv import load_dotenv

load_dotenv()

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

