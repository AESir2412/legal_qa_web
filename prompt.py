role_system = """Bạn là một chuyên gia pháp lý chuyên về luật Việt Nam, với kiến thức sâu rộng về các quy định pháp luật và ứng dụng thực tiễn của chúng. Nhiệm vụ của bạn là trả lời các câu hỏi pháp lý một cách chính xác, rõ ràng và chuyên nghiệp.  

### Hướng dẫn:  
1. Bắt buộc phải trả lời bằng tiếng Việt.  
2. Chỉ dựa trên thông tin được cung cấp để trả lời, tránh đưa ra giả định hoặc sử dụng kiến thức bên ngoài.  
3. Đảm bảo câu trả lời có cấu trúc rõ ràng, ngắn gọn và phù hợp với câu hỏi.  
4. Nếu không thể trả lời câu hỏi dựa vào các điều luật trên, hãy phản hồi rằng thông tin hiện tại chưa đủ để đưa ra câu trả lời chính xác. Nếu trả lời được, hãy nói rõ căn cứ ở đâu, văn bản nào, kèm theo trích dẫn nội dung sử dụng để trả lời trước khi trả lời
5. Không thêm tiền tố (như "Trả lời:") trước câu trả lời.  
"""

role_user = """Dưới đây là câu hỏi và thông tin liên quan để trả lời câu hỏi này  

### Câu hỏi:  
{question}  

### Thông tin liên quan:  
{context}  
"""

def create_context(relevant_docs):
    context = ""
    for i, re_law in enumerate(relevant_docs):
        title_Law = re_law["title"]
        title_Chapter = re_law["chapter_title"]
        title_Section = re_law["section_title"]
        title_Article = re_law["article_title"]
        content_Article = re_law["content"]

        context += f"""-------------------
- Tiêu đề của văn bản chứa điều luật: {title_Law}
- Tiêu đề của chương chứa điều luật: {title_Chapter} 
- Tiêu đề của mục chứa điều luật: {title_Section} 
- Tiêu đề của điều luật: {title_Article}
- Nội dung của điều luật: {content_Article}
"""
    
    return context


def create_message(relevant_docs, user_question):
    context = create_context(relevant_docs)
    prompt = role_user.replace("{question}", user_question).replace("{context}", context)
    message = [
        {
            "role": "system",
            "content": role_system
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    return message