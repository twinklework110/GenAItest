import streamlit as st
from io import StringIO
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pyperclip

load_dotenv()

def work_anniversary_message(text_data, system_message_content, vector_store):
    prompt = f"{system_message_content}\nPlease draft an good email: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return qa({"query": prompt})

def app():
    st.subheader("Choose the Right Words to Celebrate Your Coworkers’ Work Milestones")

    # Text input for Name
    name = st.text_input("Enter the name")

    # Number input for Number of years worked
    years_worked = st.number_input("Number of years worked", min_value=0, step=1)

    # Text input for Achievements
    achievements = st.text_area("Enter the achievements")

    # Selector for Relationship
    # relationship_options = ["Colleague", "Manager", "Subordinates"]
    # relationship = st.selectbox("Relationship", relationship_options)

    # Selector for Tone
    tone_options = ["Formal", "Neutral", "Informal"]
    tone = st.selectbox("Tone", tone_options)

    # Submit button
    if st.button("Generate Message"):
        if not name.strip() or not tone.strip() or not str(years_worked).strip() or not  achievements.strip():
            st.error("Please provide all the details.")
        else:
            # Convert inputs to string
            string_data = f"Name: {name}\nTone: {tone}\nYears Worked: {years_worked}\nAchievements: {achievements}"

            # Text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
            )

            texts = text_splitter.split_text(string_data)

            # OpenAI embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

            # Vector store
            vector_store = FAISS.from_texts(texts, embeddings)

            # Access system message content
            # system_message_content = "You are a highly skilled AI trained in generating work anniversary message. Please read the given details like name, number of worked years, relationship, tone and draft a good work anniversary message"
            system_message_content ="As an adept AI specialized in crafting work anniversary messages, I kindly request you to review the provided information including the individual's name, numbers of year worked,desired tone, notable achievements, and to compose a heartfelt work anniversary message. Feel free to incorporate uplifting emojis .Please compose an email based on tone. For instance, if tone selected is Formal then draft formal email."
            
            # Generate work anniversary message
            result = work_anniversary_message(string_data, system_message_content, vector_store)
            st.write("**Output:**")
            otp= result["result"]
            # st.write(result["result"])

            # if st.button("Copy"):
            #     pyperclip.copy(otp)  # Copy text to clipboard using pyperclip
            #     st.success("Message copied to clipboard!")

            if st.button("Download Message"):
             message_content = otp.encode("utf-8")  # Encode for byte data
             st.download_button(label="Download Message", data=message_content, file_ext="txt")

# Call the app function to execute it
if __name__ == '__main__':
    app()
