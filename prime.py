import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("PRIME Ai: Your Chat Assistance 🥷")
btn1 = st.button("Learner Section 💠")
btn2 = st.button("Want to Enroll 🫵")
if btn1:
    create_vector_db()

question = st.text_input("How Can I Help Youu..: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("I am Happy To Help You")
    st.write(response["result"])




