from langchain_openai import ChatOpenAI
from app.config.settings import OPENAI_MODEL


chat_model = ChatOpenAI(model=OPENAI_MODEL,
                         temperature=0,max_tokens=None,
                         timeout=None,
                         max_retries=2,
                         )
