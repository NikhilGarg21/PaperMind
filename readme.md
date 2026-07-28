# 📚 PaperMind
### Your research papers, made conversational.

🔗 **Live Demo:** [Add Streamlit App Link Here]

PaperMind is an AI-powered research paper assistant that allows users to upload research papers and interact with them using natural language. It uses Retrieval-Augmented Generation (RAG) to answer questions, summarize content, and retrieve relevant sections from documents.

---

## 🚀 Features

- 📄 Upload multiple research papers (PDF)
- 💬 Chat with your research documents
- 🔍 Semantic search using embeddings
- 📝 Context-aware answers using RAG
- 📚 Retrieved source chunks with page references
- ⚡ Fast similarity search with FAISS
- 🤖 LLM-powered responses using Mistral

---

## 🏗️ Architecture

```
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Database
    ↓
Similarity Search
    ↓
Relevant Context Retrieval
    ↓
Mistral LLM
    ↓
Final Answer
```

---

## 🛠️ Tech Stack

**Frontend**
- Streamlit

**AI / Backend**
- Python
- LangChain
- Mistral LLM
- FAISS

**Document Processing**
- PyMuPDF
- RecursiveCharacterTextSplitter

**Embeddings**
- Sentence Transformers
- all-MiniLM-L6-v2

---

## 📂 Project Structure

```
PaperMind/
│
├── app.py              # Streamlit UI
├── read_pdf.py         # PDF processing and vector creation
├── chat.py             # RAG question answering pipeline
├── prompts.py          # LLM prompts
│
├── data/               # Uploaded PDFs
├── vectorstore/        # FAISS index
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PaperMind.git
cd PaperMind
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
MISTRAL_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Upload research papers.
2. Extract text from PDFs.
3. Split documents into meaningful chunks.
4. Convert chunks into vector embeddings.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks for user questions.
7. Generate answers using the LLM.

---

## 🔮 Future Improvements

- Support more document formats
- Better citation highlighting
- Multi-paper comparison
- Persistent user document storage
- Conversation memory

---


