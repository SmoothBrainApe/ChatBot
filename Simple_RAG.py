from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from Embed import embed_model

# Create Embeddings
embeddings = OllamaEmbeddings(model=embed_model, show_progress=False)

# Setup Chroma database
db = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)

# Create retriever
retriever = db.as_retriever(
    search_type='similarity',
    search_kwargs= {'k': 5}
)

# Initialize Ollama Model
model = 'Seren'

llm = ChatOllama(
    model=model,
    keep_alive="3h",
    max_tokens=1024,
)
