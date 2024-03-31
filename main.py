import streamlit as st
from Langchain_Retrieve import get_qa_chain, create_vector_db

st.title("PRIME Ai: Your Chat Assistance 🥷")
btn = st.button("Devoloper Section 💠")
if btn:
    create_vector_db()

question = st.text_input("How Can I Help You... 🖊️")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("I am Happy To Help You 🥷")
    st.write(response["result"])