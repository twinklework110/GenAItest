import streamlit as st
from io import StringIO
import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from io import BytesIO
import docx2txt 
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

load_dotenv()

def generate_minutes(text_data, system_message_content, vector_store):
    prompt = f"{system_message_content}\nPlease generate statements indicating who said what based on the provided text: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        # llm=OpenAI(model_name="text-davinci-003"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please upload a .txt file to generate who said what from meeting notes")

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx"])

    # Enable button only if file is uploaded
    if uploaded_file is not None:
        submit_button = st.button('Generate Output')
    else:
        submit_button = None
        
    if submit_button:
        with st.spinner("Processing"):
        
         file_content = uploaded_file.read()
        # To convert to a string based IO:
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

        # Access system message content (assuming first element is the system message)
        #system_message_content = "You are an AI assistant which generates sentiment analysis on given text data."
        system_message_content = "As an AI proficient in language comprehension and speaker attribution, your role entails accurately identifying speakers and their corresponding statements within the provided meeting script in summarized way. Consider the context of the conversation, including any contextual clues, speech patterns, or verbal cues that may distinguish speakers. Your task is to precisely attribute each statement to the appropriate speaker. Craft responses that clearly indicate who said what, ensuring accuracy and clarity in your identification process. Dont just take statements as is from the provided text but also briefly summarized.Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text."
        result = generate_minutes(text_data, system_message_content, vector_store)  # Pass vector_store as argument
        st.write("**Output:**")
        st.write(result["result"])
    else:
        None

# Call the app function to execute it
if __name__ == '__main__':
    app()
