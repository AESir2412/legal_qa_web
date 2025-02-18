from prompt import create_message

def generate_answer(client, relevant_docs, user_question):
    messages = create_message(relevant_docs, user_question)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )
    
    print("Done QA!")
    return chat_completion.choices[0].message.content.strip()
