from langchain_community.embeddings import OpenAIEmbeddings

from env_utils import QWEN_BASE_URL, QWEN_API_KEY, OPENAI_API_KEY

# EMBEDDING = OpenAIEmbeddings(
#     base_url=QWEN_BASE_URL,
#     api_key=QWEN_API_KEY,
#     model="text-embedding-v3"
# )

EMBEDDINGS = OpenAIEmbeddings(
    base_url=OPENAI_API_KEY,
)
