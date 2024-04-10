# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import pyperclip

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
    st.markdown(
        """
        <style>
        div[data-testid="stForm"]{
            position: fixed;
            bottom: 0;
            width: 50%;
            background-color: #2b313e;
            padding: 10px;
            height: 195px;
            z-index: 10;
        }
        </style>
        """, unsafe_allow_html=True
    )

    chat = ChatOpenAI(temperature=0)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("People Success GPT 🤖")

    # sidebar with user input
    # with st.sidebar:
    with st.container():
        myfrom=st.form(key="form",clear_on_submit=True)
        user_input = myfrom.text_area("Ask your question: ", key='user_input',value='')

    with st.container(): 
        if myfrom.form_submit_button("Submit"):
        # handle user input
            if user_input:
              st.session_state.messages.append(HumanMessage(content=user_input))
              with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
              st.session_state.messages.append(
                AIMessage(content=response.content))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')
            copy_button_id = f"copy_button_{i}"
            if st.button("Copy", key=copy_button_id):
               copy_to_clipboard(msg.content)

def copy_to_clipboard(content):
    pyperclip.copy(content)

if __name__ == '__app__':
    app()