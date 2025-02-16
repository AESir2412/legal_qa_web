import streamlit as st
import os 
from retrieval import bert_ensemble
from qa import generate_answer

def main():
    st.title("Legal Question Answering System")  
    st.write("Ask any legal question and get answers based on relevant legal documents.")

    # Create input text area for user question
    user_question = st.text_area("Enter your legal question:", height=100)

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Retrieving relevant documents..."):
                # Retrieve relevant documents using bert_ensemble
                relevant_docs = bert_ensemble(user_question)
            
            if relevant_docs:
                # Show relevant documents
                st.subheader("Relevant Legal Articles:")
                for doc in relevant_docs:
                    # Create clickable title with href
                    title = f"Điều {doc['article_id']} {doc['title']}" if doc['title'] else f"Điều {doc['article_id']}"
                    st.markdown(f"[{title}]({doc['href']})")

                with st.spinner("Generating answer..."):
                        # Generate and display the answer
                        answer = generate_answer(relevant_docs, user_question)
                        st.subheader("Answer:")
                        st.write(answer)
            else:
                st.error("No relevant documents found. Please try rephrasing your question.")
        else:
            st.warning("Please enter a question.")

    # Add footer
    st.markdown("---")
    st.markdown("*This is a legal question answering system powered by GPT and document retrieval.*")

if __name__ == "__main__":
    main()

