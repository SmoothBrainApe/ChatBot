from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

embed_model = 'snowflake-arctic-embed'

# Load document from directory
loader = DirectoryLoader('./Docs/', glob='**/*.txt')

print('document loaded')
documents = loader.load()
print(len(documents))

# Create embeddings
embeddings = OllamaEmbeddings(model=embed_model, show_progress=True)

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

# Split documents into chunks
texts = text_splitter.split_documents(documents=documents)

# Create Vector Store
if os.path.exists('chroma_db/chroma.sqlite3'):
    pass
else:
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory='./chroma_db'
    )
    print('Vector Store created!')
