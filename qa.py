from prompt import create_message
import time

def generate_answer(client, relevant_docs, user_question):
    start_time = time.time()  
    messages = create_message(relevant_docs, user_question)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )

    end_time = time.time() 
    print(f"Done QA! Time taken: {end_time - start_time:.2f} seconds")
    return chat_completion.choices[0].message.content.strip()
