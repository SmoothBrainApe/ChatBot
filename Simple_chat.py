from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = '''
    Here is the conversation history: {context}
    User: {user}
    Seren:
    '''


model = OllamaLLM(model='Seren')
prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | model


def conversation():
    context = ''
    print('Press q to quit')
    while True:
        user_input = input('User: ')
        if user_input.lower() == 'q':
            break
        result = chain.invoke({'context': context, 'user': user_input})
        print(f'Seren: {result}')
        context += f'\nUser: {user_input}\nSeren: {result}'


if __name__ == '__main__':
    conversation()
