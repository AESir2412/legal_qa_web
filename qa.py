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


def stream_generate_answer(client, relevant_docs, user_question):
    start_time = time.time()
    
    messages = create_message(relevant_docs, user_question)

    # Gọi API với chế độ streaming
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        stream=True,  # Bật chế độ streaming
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content  # Trả về từng phần của câu trả lời

    end_time = time.time()
    print(f"Done QA! Time taken: {end_time - start_time:.2f} seconds")