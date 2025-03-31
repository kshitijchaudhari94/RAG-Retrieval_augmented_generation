import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load Mistral API key from environment
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    st.error("❌ MISTRAL_API_KEY not found. Please add it to your .env file.")
    st.stop()


# Step 1: Extract and Chunk PDF Text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page_text := page.extract_text():
                text += page_text
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return splitter.split_text(text)

# Step 2: Vector Embedding and FAISS Index

def get_vector_store(text_chunks):
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Step 3: RAG Chain (Prompt + Memory + Retriever)

def get_conversational_chain(vector_store):
    model = ChatMistralAI(mistral_api_key=api_key)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    ----------------
    {context}
    ----------------
    Question: {question}
    Answer:
    """)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Step 4: Streamlit UI

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, msg in enumerate(st.session_state.chat_history):
        role = "👤 You" if i % 2 == 0 else "🤖 Bot"
        st.markdown(f"**{role}:** {msg.content}")

def main():
    st.set_page_config("Mistral OpenSource LLM Model RAG Chatbot")
    st.title("HR RAG Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a Question:")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload HR Policy Documents", accept_multiple_files=True)
        if st.button("⚙️ Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("✅ Ready to chat!")
            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
