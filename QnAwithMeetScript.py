import streamlit as st
import os
from io import BytesIO
import docx2txt  # Import the docx2txt library

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def generate_minutes(text_data, user_question, system_message_content, vector_store):
    prompt = f"{system_message_content}\nUser Question: {user_question}\nPlease answer the user question: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please upload a .txt or .docx file to chat with the document.")

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx"])

    #with st.sidebar:
    # Chat option
    user_question = st.text_input("Ask a question about the document:")

    # Enable button only if file is uploaded
    if uploaded_file is not None:
        submit_button = st.button('Submit')
    else:
        submit_button = None

    if submit_button:
        # Read the uploaded file
        file_content = uploaded_file.read()
        
        if uploaded_file.type == "text/plain":  # If file is .txt
            text_data = file_content.decode("utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # If file is .docx
            docx_data = BytesIO(file_content)
            text_data = docx2txt.process(docx_data)
        else:
            st.error("Unsupported file type. Please upload a .txt or .docx file.")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
        )

        texts = text_splitter.split_text(text_data)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        vector_store = FAISS.from_texts(texts, embeddings)

        # Access system message content
        system_message_content = "As an AI assistant, your task is to provide accurate responses to user questions. Please ensure precision in your answers, and if you're uncertain or lack information, it's appropriate to indicate so rather than providing incorrect responses." 

        result = generate_minutes(text_data, user_question, system_message_content, vector_store)
        st.write("**Output:**")
        st.write(result["result"])
    # else:
    #     st.warning("Please upload a document to proceed.")

# Call the app function to execute it
if __name__ == '__main__':
    app()
