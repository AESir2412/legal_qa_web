import openai
from constant import *
import os

def generate_answer(relevant_docs, user_question):
   # Construct the prompt for the GPT model
    prompt = "Dựa vào những điều luật sau:\n"
    for doc in relevant_docs:
        prompt += f"- Điều {doc['article_id']} {doc['title']}: {doc['content']}\n"
    
    prompt += f"\nHãy trả lời câu hỏi sau: {user_question}\n"

    # Extra requirments
    prompt += """Nếu không thể trả lời câu hỏi dựa vào các điều luật trên, hãy phản hồi rằng thông tin hiện tại chưa đủ để đưa ra 
                 câu trả lời chính xác. Nếu trả lời được, hãy nói rõ căn cứ ở đâu, văn bản nào trước khi trả lời\n"""

    client = openai.OpenAI(
        api_key=API_KEY
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )

    answer = chat_completion.choices[0].message.content.strip()

    print("Done QA!")
    return answer

if __name__ == "__main__":
    main()