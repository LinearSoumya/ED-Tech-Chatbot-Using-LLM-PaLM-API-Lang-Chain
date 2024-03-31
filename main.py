import streamlit as st
from Langchain_Retrieve import get_qa_chain, create_vector_db

st.title("PRIME Ai: Your Chat Assistance 🥷")

# Create a button to trigger database creation
btn = st.button("Developer Section 💠")
if btn:
    create_vector_db()

# Increase the height of the text input for the question
question = st.text_input("How Can I Help You... 🖊️")

# Button to trigger the display of output
if st.button("Get Answer ⬇️"):
    if question:
        chain = get_qa_chain()
        response = chain(question)

        st.header("I am Happy To Help You 🥷")
        st.write(response["result"])
