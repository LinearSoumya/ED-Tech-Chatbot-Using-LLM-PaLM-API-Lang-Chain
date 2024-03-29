import streamlit as st
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import PromptTemplate, RetrievalQA

# Initialize LLM model
api_key = "YOUR_GOOGLE_API_KEY"
llm = GooglePalm(google_api_key=api_key, temperature=0.9)

# Load data
loader = CSVLoader(file_path="ED-Tech.csv", source_column="prompt", encoding="latin-1")
data = loader.load()

# Initialize instructor embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="google/flan-t5-xl")

# Create vector database
vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

# Create retriever
retriever = vectordb.as_retriever(score_threshold=0.7)

# Define prompt template
prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# Initialize RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

# Streamlit app
st.title("PRIME Ai: Your Chat Assistance")

def generate_response(question):
    response = chain(question)
    return response

# User interface
user_question = st.text_input("Enter your question:")
if st.button("Get Response"):
    if user_question:
        response = generate_response(user_question)
        st.write("Response:", response)
    else:
        st.write("Please enter a question.")





