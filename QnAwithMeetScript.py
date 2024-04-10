import streamlit as st
import os
from io import BytesIO
import docx2txt  # Import the docx2txt library

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import pyperclip
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()

ss = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 5%;
}
.chat-message .avatar img {
  max-width: 20px;
  max-height: 20px;
  border-radius: 10%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)

if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

def generate_minutes(text_data, user_question, system_message_content, vector_store):
    prompt = f"{system_message_content}\nUser Question: {user_question}\nPlease answer the user question: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
        memory=memory
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please upload a .txt or .docx file to chat with the document.")

     # Container with background color for file upload option
    st.markdown(
        """
        <style>
        .file-uploader-container {
            background-color: #ee1515;
            padding: 1rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
   
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

        st.session_state.conversation = generate_minutes(text_data, user_question, system_message_content, vector_store)
        st.write("**Output:**")
        # st.write(result["result"])

        if user_question:
           response = st.session_state.conversation({'question': user_question})
           st.session_state.chat_history = response.get('chat_history', [])

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
         st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
         st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
         copy_button_id = f"copy_button_{i}"
         if st.button("Copy", key=copy_button_id):
            copy_to_clipboard(message.content)

def copy_to_clipboard(content):
    pyperclip.copy(content)
                     
    
        # if st.button("Copy"):
        #     st.write(f'<script>navigator.clipboard.writeText("{result["result"]}")</script>', unsafe_allow_html=True)

# Call the app function to execute it
if __name__ == '__main__':
    app()
