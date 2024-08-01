import ollama


model = 'Seren'
# message_history = {}


def chat(message):
    messages = [{
        'role': 'user',
        'content': message,
        }]
    response = ollama.chat(model=model, messages=messages, stream=True)
    for line in response:
        print(line['message']['content'], end='', flush=True)
    # message_history.append(answer)


while True:
    # print(message_history)
    print('\nQ to quit')
    prompt = input('Enter your message: ')
    if prompt.lower() == 'q':
        break
    else:
        chat(prompt)
