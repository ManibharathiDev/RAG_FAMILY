import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load and split the documents
loader = TextLoader('data/family.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embed and store in vector DB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Build the RAG pipeline using LCEL
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate.from_template(
    """Answer the question based on the context below.

Context:
{context}

Question: {question}
"""
)

rag_chain = (
    {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG Chatbot")

query = st.text_input("Ask a question:", placeholder="e.g. Who is Manibharathi")

if query:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(query)
    st.markdown("### 💬 Response")
    st.write(response)
