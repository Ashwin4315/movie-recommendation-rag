import streamlit as st
from rag_pipeline import rag_answer

st.title("🎬 Movie Review RAG Assistant (Groq Edition)")

question = st.text_input("Ask something about any movie review:")

if st.button("Ask"):
    if question.strip():
        answer, sources = rag_answer(question)

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Top Retrieved Review Evidence:")
        for src in sources:
            st.write("- " + src)
