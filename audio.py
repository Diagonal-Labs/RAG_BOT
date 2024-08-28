import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import speech_recognition as sr
import pyttsx3
import asyncio

# Setup logging
logging.basicConfig(level=logging.DEBUG)

DB_FAISS_PATH = 'vectorstore/db_faiss'

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    logging.debug("Setting custom prompt.")
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    logging.debug("Initializing RetrievalQA chain.")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=False, chain_type_kwargs={'prompt': prompt})
    return qa_chain

def load_llm():
    logging.debug("Loading the LLM model.")
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature=0.5)
    return llm

def qa_bot():
    logging.debug("Initializing the QA bot.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def speech_to_text():
    logging.debug("Converting speech to text.")
    with sr.Microphone() as source:
        logging.debug("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            logging.debug(f"Recognized: {query}")
            return query
        except sr.UnknownValueError:
            logging.error("Could not understand the audio.")
            return "Sorry, I did not understand that."
        except sr.RequestError as e:
            logging.error(f"Request error: {e}")
            return f"Could not request results; {e}"

def text_to_speech(text):
    logging.debug("Converting text to speech.")
    tts_engine.say(text)
    tts_engine.runAndWait()

@cl.on_chat_start
async def start():
    logging.debug("Starting the bot.")
    try:
        chain = qa_bot()

        # Bot introduction
        intro_text = "Hi, Welcome to the Medical Bot. Let's start our conversation. You can say 'Bye' anytime to stop."
        text_to_speech(intro_text)
        await cl.Message(content=intro_text).send()

        cl.user_session.set("chain", chain)

        # Continuous conversation loop
        while True:
            # Activate microphone automatically
            user_input = speech_to_text()
            await cl.Message(content=f"You said: {user_input}").send()

            # Check if the user wants to end the conversation
            if "bye" in user_input.lower():
                goodbye_text = "Goodbye! Have a great day!"
                text_to_speech(goodbye_text)
                await cl.Message(content=goodbye_text).send()
                break

            # Process the input through the chain and respond
            res = await chain.acall(user_input)
            answer = res["result"]

            # Respond in both text and speech
            await cl.Message(content=answer).send()  # Send text response first
            text_to_speech(answer)  # Convert text to speech after sending the text

    except Exception as e:
        logging.error(f"Error during bot start: {e}")
        await cl.Message(content="An error occurred. Please try restarting the bot.").send()

@cl.on_message
async def main(message: cl.Message):
    logging.debug(f"Received message: {message.content}")
    try:
        chain = cl.user_session.get("chain")

        # Check if the user wants to use voice input
        if message.content.lower() == "speak":
            user_input = speech_to_text()
            await cl.Message(content=f"You said: {user_input}").send()
        else:
            user_input = message.content
        
        res = await chain.acall(user_input)
        answer = res["result"]

        # Convert the answer to speech if the user requested speech output
        await cl.Message(content=answer).send()  # Send text response first
        text_to_speech(answer)  # Convert text to speech after sending the text

    except Exception as e:
        logging.error(f"Error during message processing: {e}")
        await cl.Message(content="An error occurred during message processing.").send()
