import streamlit as st
import os 
from retrieval import bert_ensemble
from qa import generate_answer

def main():
    st.title("Hệ thống Hỏi đáp pháp luật tự động")  
    st.write("Hãy đưa ra bất cứ câu hỏi pháp lý nào, chúng tôi sẽ trả lời dựa trên các điều luật liên quan.")

    # Without Reset button ----------------------------------------------------------------------------------
    # Create input text area for user question
    user_question = st.text_area("Điền câu hỏi pháp lý tại đây:", height=100)

    if st.button("Trả lời"):
        if user_question:
            with st.spinner("Tìm kiếm điều luật liên quan..."):
                # Retrieve relevant documents using bert_ensemble
                relevant_docs = bert_ensemble(user_question)
            
            if relevant_docs:
                with st.spinner("Tạo câu trả lời..."):
                    # Generate and display the answer
                    answer = generate_answer(relevant_docs, user_question)
                    st.subheader("Trả lời:")
                    st.write(answer)

                # Show relevant documents
                st.subheader("Điều luật liên quan:")
                for doc in relevant_docs:
                    # Create clickable title with href
                    title = f"Điều {doc['article_id']} {doc['title']}" if doc['title'] else f"Điều {doc['article_id']}"
                    st.markdown(f"[{title}]({doc['href']})")
                    
                    # Add content in a dropdown
                    with st.expander("Xem nội dung"):
                        st.write(doc['content'])
            else:
                st.error("Không tìm thấy nội dung liên quan nào. Hãy thử xem lại nội dung câu hỏi của bạn.")
        else:
            st.warning("Vui lòng điền câu hỏi của bạn.")


    # # Change (with reset button) -------------------------------------------------------------------
    # # CURRENTLY BUGGED (Reset 2 lần thì nó hiện nút reset cũ)
    # if 'user_question' not in st.session_state:
    #     st.session_state.user_question = ''
    # if 'relevant_docs' not in st.session_state:
    #     st.session_state.relevant_docs = None

    # # Create input text area for user question
    # user_question = st.text_area("Enter your legal question:", height=100, value=st.session_state.user_question)

    # if st.button("Get Answer"):
    #     if user_question:
    #         st.session_state.user_question = user_question
    #         with st.spinner("Retrieving relevant documents..."):
    #             # Retrieve relevant documents using bert_ensemble
    #             st.session_state.relevant_docs = bert_ensemble(user_question)
            
    #         if st.session_state.relevant_docs:
    #             # Show relevant documents
    #             st.subheader("Relevant Legal Articles:")
    #             for doc in st.session_state.relevant_docs:
    #                 # Create clickable title with href
    #                 title = f"Điều {doc['article_id']} {doc['title']}" if doc['title'] else f"Điều {doc['article_id']}"
    #                 st.markdown(f"[{title}]({doc['href']})")
                    
    #                 # Add content in a dropdown
    #                 with st.expander("Show Content"):
    #                     st.write(doc['content'])

    #             with st.spinner("Generating answer..."):
    #                 # Generate and display the answer
    #                 answer = generate_answer(st.session_state.relevant_docs, st.session_state.user_question)
    #                 st.subheader("Answer:")
    #                 st.write(answer)
    #         else:
    #             st.error("No relevant documents found. Please try rephrasing your question.")
    #     else:
    #         st.warning("Please enter a question.")


    # # Add a reset button
    # if st.button("Reset"):
    #     st.session_state.user_question = ''
    #     st.session_state.relevant_docs = None
    #     st.experimental_rerun()

    # Add footer
    st.markdown("---")
    st.markdown("*Hệ thống Hỏi đáp pháp luật tự động tạo bởi nhóm BabyFour.*")

if __name__ == "__main__":
    main()

