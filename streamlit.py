import streamlit as st
import os
from retrieval import bert_ensemble
from qa import generate_answer
import openai
from dotenv import load_dotenv
# from bm25 import _load_bm25_model, _load_article_info
from bm25 import _load_bm25s_retriever, _load_article_info

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Hàm load model có cache để tránh load lại nhiều lần
@st.cache_resource
def load_models():
    print("⚡ Đang khởi động hệ thống...")

    print("🔄 Đang load BM25 model...")
    # bm25_model = _load_bm25_model()
    bm25s_retriever = _load_bm25s_retriever()
    print("✅ BM25 model loaded!")

    print("🔄 Đang load Article Info...")
    article_info = _load_article_info()
    print("✅ Article Info loaded!")

    print("🔄 Đang khởi tạo OpenAI Client...")
    client = openai.OpenAI(api_key=API_KEY)
    print("✅ OpenAI Client initialized!")

    print("🚀 Hệ thống đã sẵn sàng!")

    # return bm25_model, article_info, client
    return bm25s_retriever, article_info, client

# bm25_model, article_info, client = load_models()
bm25s_retriever, article_info, client = load_models()

# Bắt đầu Streamlit UI
def main():
    st.title("Hệ thống Hỏi đáp pháp luật tự động")  
    st.write("Hãy đưa ra bất cứ câu hỏi pháp lý nào, chúng tôi sẽ trả lời dựa trên các điều luật liên quan.")

    user_question = st.text_area("Điền câu hỏi pháp lý tại đây:", height=100)

    if st.button("Trả lời"):
        if user_question:
            with st.spinner("Tìm kiếm điều luật liên quan..."):
                print("🔍 Đang tìm kiếm điều luật liên quan...")
                relevant_docs = bert_ensemble(bm25s_retriever, article_info, user_question)

            if relevant_docs:
                with st.spinner("Tạo câu trả lời..."):
                    answer = generate_answer(client, relevant_docs, user_question)
                    st.subheader("Trả lời:")
                    st.write(answer)

                st.subheader("Điều luật liên quan:")
                for doc in relevant_docs:
                    title = f"Điều {doc['article_id']} {doc['title']}" if doc['title'] else f"Điều {doc['article_id']}"
                    st.markdown(f"[{title}]({doc['href']})")
                    with st.expander("Xem nội dung"):
                        st.write(doc['article_title'] + "\n" + doc['content'])
            else:
                st.error("Không tìm thấy nội dung liên quan nào. Hãy thử xem lại nội dung câu hỏi của bạn.")
        else:
            st.warning("Vui lòng điền câu hỏi của bạn.")

    st.markdown("---")
    st.markdown("*Hệ thống Hỏi đáp pháp luật tự động tạo bởi nhóm BabyFour.*")

if __name__ == "__main__":
    main()
