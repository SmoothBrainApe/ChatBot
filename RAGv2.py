from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# Create Embeddings
embeddings = OllamaEmbeddings(model='snowflake-arctic-embed:335m', show_progress=False)
print('Embedding loaded')

# Setup Chroma database
db = Chroma(persist_directory='./Vector/chroma_db', embedding_function=embeddings)
print('Database loaded')

# Create retriever
retriever = db.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 6}
)
print("Retriever initialized")

# Initialize Ollama Model
model = 'Seren'

llm = OllamaLLM(
    model=model,
    max_tokens=1024,
    keep_alive='3h'
)
print('Model loaded')


# Create prompt and template
system_prompt = '''
    Talk to the user with both context and chat history in mind.
    Do not repeat the chat history or the context. Just provide the answer.
    '''

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{question}'),
    ]
)


# Create RAG Chain
chain = prompt | llm | StrOutputParser()

chain.invoke(
        {
            'chat_history': [
                HumanMessage(content=''),
                AIMessage(content=''),
            ],
            'question': '',
        }
    )

chat_system_prompt = '''
            Use the context provided for answering the question.
            {context}
        '''

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', chat_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{question}'),
    ]
)


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def qa_prompt(input: dict):
    if input.get('chat_history'):
        return chain
    else:
        return input['question']


rag_chain = (
    RunnablePassthrough.assign(
        context=qa_prompt | retriever | format_docs
    )
    | chat_prompt
    | llm
)

chat_history = []


def chat(message):
    question = message
    ai_msg = rag_chain.invoke(
        {
            'question': question,
            'chat_history': chat_history
        }
    )
    return ai_msg


if __name__ == '__main__':
    print('Start the chat. Enter q to quit')
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'q':
            break
        else:
            output = chat(user_input)
            print(f'Seren: {output}')
            chat_history.extend(
                [
                    HumanMessage(content=question), output
                ]
            )
