from langchain_openai import OpenAIEmbeddings
from app.config.settings import EMBEDDING_MODEL


underlying_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)