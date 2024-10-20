import os
import json
import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from vectorize_documents import embeddings

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to retrieve answer from the question-answering model
def get_answer(question, context):
    qa_result = qa_pipeline(question=question, context=context)
    return qa_result.get('answer')

# Setting up vectorstore for document retrieval
def setup_vectorstore():
    persist_directory = os.path.join(os.path.dirname(__file__), "vector_db_dir")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

# Setting up fallback LLaMA chain
def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)
    return chain

# Streamlit UI setup
st.set_page_config(
    page_title="Question Answering ChatBot",
    page_icon="📚",
    layout="centered"
)

# Title at the top
st.title("📚 Question Answering ChatBot")

# Add a robo-themed GIF (you can replace this URL with any GIF URL you prefer)
# Display centered GIF
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZTJxcndrYW1pcnJ4bXpxNWM5eGYwa3J6d3BlNzQ4NnJsaXczYmZ4YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/58OujxlE7e19Mjv0gj/giphy.webp" alt="GIF" width="300" height="300">
    </div>
    """, unsafe_allow_html=True
)

# Input area for user-provided context
context_input = st.text_area("Context (optional):", placeholder="Provide context if available")

# "Ask AI" chat input field immediately after the context
user_input = st.text_input("Ask AI:")

# Check for user input
if user_input:
    # If user provides context in the context text area, use it
    if context_input:
        context = context_input
        answer = get_answer(user_input, context)
    else:
        # If no answer is found, fallback to LLaMA 2
        llama_chain = chat_chain(setup_vectorstore())
        llama_response = llama_chain.invoke({"query": user_input})
        answer = llama_response["result"]

    # Display the answer
    st.write("Answer:", answer)

