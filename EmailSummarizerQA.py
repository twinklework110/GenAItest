# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import pyperclip
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    #st.set_page_config(
     #   page_title="TA ChatGPT",
      #  page_icon="🤖"
    #)


def app():
    init()

    chat = ChatOpenAI(temperature=0)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a highly skilled AI trained in email comprehension and summarization. I would like you to read the given email and summarize it into a concise abstract paragraph.Please consider instructions provided in QnA text_area and summarize email accordingly. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire email. Please avoid unnecessary details or tangential points.Please follow the specified sequence each in new line with bullet points : 1) Recipients, 2) Sender, 3) Subject, and 4) Email Body 5) Next Step.")
        ]

    st.header("Email Summarizer 📩")
    

    def on_click():
        st.session_state['user_input'] = ""

    
    # sidebar with user input
    with st._bottom:
        myfrom=st.form(key="form",clear_on_submit=True)
        user_input = myfrom.text_area("Provide email to summarize: ", key='user_input',value='')

    with st._bottom: 
        if myfrom.form_submit_button("Submit"):
            # handle user input
            if user_input:
                st.session_state.messages.append(HumanMessage(content=user_input))
                with st.spinner("Thinking..."):
                    response = chat(st.session_state.messages)
                st.session_state.messages.append(
                    AIMessage(content=response.content,example=False))
    
  
    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, )
        else:
            message(msg.content, is_user=False,)



if __name__ == '__main__':
    app()
