from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
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
    search_kwargs={'k': 5}
)

# Initialize Ollama Model
model = 'Seren'

llm = ChatOllama(
    model=model,
    keep_alive="3h",
    max_tokens=1024,
)

# Create prompt and template
template = '''
    Answer the questions from the context given and history of chat.
    Here is the chat history: {history}
    Here is the context: {context}
    User: {question}
    Seren:
    '''

prompt = ChatPromptTemplate.from_template(template)

# Create RAG Chain
rag_chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | prompt
    | llm
)


# Function for RAG and conversation
def conversation():
    print('\nSeren:', end=' ', flush=True)
    for chunk in rag_chain.stream(user_input):
        print(chunk.content, end='', flush=True)
    print('')


if __name__ == '__main__':
    print('Enter q to quit')
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'q':
            break
        else:
            conversation()
