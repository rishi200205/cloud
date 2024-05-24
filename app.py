# %%
import os
import streamlit as st
import dotenv
from dotenv import load_dotenv
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')

# %%
from langchain_google_genai import ChatGoogleGenerativeAI

safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,}
# %%
llm = ChatGoogleGenerativeAI(model = 'gemini-pro', google_api_key = google_api_key, safety_settings=safety_settings)
#llm.invoke("Write top 10 most populous countries")

# %%
# %%
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# %%
from langchain.prompts import ChatPromptTemplate

template = '''You are a mental health chatbot. Impersonate a Helpful person. You are trained in making people understand their emotions better, and helping them deal with negative emotions. You are patient, caring, understanding, and love to help people. People can always trust me. If You make mistakes, You apologize and try to correct them. You really care about helping people work with their emotions. You know various techniques such as CBT.
{Question}
'''
prompt = ChatPromptTemplate.from_template(template)


# %%
chain0 = prompt | llm | parser

# %%

def generate_response(text):
    response = chain0.invoke({
        'Question': text
    })

    return response


st.title("YourDost")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Sup!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

