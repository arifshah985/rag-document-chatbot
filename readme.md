#  Enterprise Document Intelligence Chatbot (RAG System)

A production-ready, local Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**, and **Streamlit**. This application allows corporate teams to securely upload enterprise documents (PDF format), index their contents into a vector database, and perform context-aware semantic search via an AI-powered conversational interface.

---

##  Key Features
* **Document Processing Pipeline:** Automates PDF text extraction and structural content chunking.
* **Vector Embeddings Engine:** Leverages OpenAI Embeddings to translate textual context into mathematical vectors.
* **Local Context Storage:** Utilizes **ChromaDB** as an in-memory vector database for high-performance context retrieval.
* **Intelligent Conversation Loop:** Implements LangChain pipelines combined with `gpt-4o-mini` to deliver zero-hallucination, context-strictly-bounded answers.

---

##  Architecture & Tech Stack
* **Language:** Python 3.12+
* **Framework:** Streamlit (UI Dashboard)
* **Orchestration:** LangChain Core & LangChain Classic
* **Vector DB:** ChromaDB
* **LLM Engine:** OpenAI API (`gpt-4o-mini`)

---

##  Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd rag-document-chatbot
   ```

2. **Install Dependencies:**
   ```bash
   pip install langchain langchain-community langchain-openai langchain-classic streamlit chromadb pypdf python-dotenv
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and securely add your credentials:
   ```text
   OPENAI_API_KEY=your_secret_openai_api_key
   ```

4. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```

---

##  Developer Status & Saudi Arabia Availability
* **Education:** MS in Software Engineering
* **Core Focus:** Python Development • AI/LLM Integration • Backend Architectures
* **Current Status:** **Locally based in Saudi Arabia with a Transferable Iqama**
* **Availability:** Open for Immediate Deployment / Ready to Join

---
*Developed as part of an advanced AI Engineering portfolio.*
