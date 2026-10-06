import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
# PASTE THESE NEW LINES INSTEAD:
# Updated imports for modern LangChain v1.0+
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from langchain_core.prompts import ChatPromptTemplate

# 1. Load API keys from .env file
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please add your OPENAI_API_KEY to the .env file.")
    st.stop()

st.set_page_config(page_title="Enterprise AI Doc Search", layout="centered")
st.title("📄 Enterprise Document Intelligence Chatbot")
st.write("Upload a corporate document/PDF and ask questions directly to the AI.")

# 2. File Upload UI
uploaded_file = st.file_uploader("Upload your PDF contract or manual", type="pdf")

if uploaded_file is not None:
    # Save the file temporarily to read it
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success("File uploaded successfully! Processing...")

    with st.spinner("Analyzing and indexing document text..."):
        # 3. Process PDF: Load and split into readable chunks
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 4. Create Embeddings & Store in a Vector DB
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    # 5. Set up the LLM Chat Chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system_prompt = (
        "You are an expert corporate assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 6. Chat interface
    user_query = st.text_input("Ask something about this document:")
    if user_query:
        with st.spinner("Searching document for answers..."):
            response = rag_chain.invoke({"input": user_query})
            st.write("### 🤖 AI Answer:")
            st.info(response["answer"])
