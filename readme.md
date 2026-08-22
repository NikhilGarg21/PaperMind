# 📚 PaperMind

### AI-Powered Document Question Answering System

🔗 **Live Demo:** https://papermind-ejnapjogvxmqkdj8muxi7r.streamlit.app/

PaperMind is an AI-powered document question answering system built using **Retrieval-Augmented Generation (RAG)**. Users can upload one or more PDF documents and interact with them through natural language queries.

The system processes documents, splits their content into chunks, generates embeddings, stores them in a FAISS vector database, retrieves relevant document sections, and uses a Mistral LLM to generate context-aware answers.

The project also incorporates **MLOps practices using YAML configuration, logging, DVC, and DVCLive** to make the RAG pipeline reproducible, configurable, and experiment-friendly.

---

## 🚀 Features

* 📄 Upload multiple research papers (PDF)
* 💬 Chat with uploaded documents
* 🔍 Semantic search using vector embeddings
* 📝 Context-aware answers using RAG
* 📚 Retrieved source chunks with page references
* ⚡ Fast similarity search using FAISS
* 🤖 LLM-powered responses using Mistral
* ⚙️ Configurable parameters using YAML
* 📋 Structured application logging
* 🔄 Reproducible data pipeline using DVC
* 🧪 Experiment tracking using DVCLive

---

## 🏗️ Architecture

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Database
    ↓
Semantic Similarity Search
    ↓
Relevant Document Retrieval
    ↓
Prompt Construction
    ↓
Mistral LLM
    ↓
Final Answer + Sources
```

### MLOps Pipeline

```text
params.yaml
    ↓
DVC Pipeline
    ↓
read_pdf.py
    ↓
vector_db/
    ↓
chat.py
    ↓
output/chat_output.txt
    ↓
DVCLive Experiments
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### AI / Backend

* Python
* LangChain
* Mistral LLM
* FAISS

### Document Processing

* PyMuPDF
* PyPDFLoader
* RecursiveCharacterTextSplitter

### Embeddings

* Sentence Transformers
* `all-MiniLM-L6-v2`

### MLOps

* YAML configuration
* Python logging
* DVC
* DVCLive
* Git / GitHub

---

## 📂 Project Structure

```text
PaperMind/
│
├── src/
│   ├── app.py                 # Streamlit application
│   ├── read_pdf.py            # PDF processing and vector creation
│   ├── chat.py                # RAG question answering pipeline
│   ├── docs_preprocess.py     # Text preprocessing utilities
│   ├── logger.py              # Logging configuration
│   └── __init__.py
│
├── data/                      # Input PDF documents
├── vector_db/                 # FAISS vector database
├── output/                    # Generated chatbot output
│
├── params.yaml                # Configurable parameters
├── dvc.yaml                   # DVC pipeline definition
├── .dvcignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/NikhilGarg21/PaperMind.git
cd PaperMind
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key
```

### 5. Run the application

```bash
streamlit run src/app.py
```

---

## ⚙️ Configuration

Project parameters are managed through `params.yaml` instead of being hard-coded.

Example:

```yaml
read_pdf:
  chunk_size: 1000
  chunk_overlap: 200

chat:
  temperature: 0
  retrieval_k: 5
  display_chunks: 3
```

This allows retrieval and document-processing parameters to be changed without modifying the application code.

---

## 🔄 DVC Pipeline

The project uses DVC to make the document-processing and chatbot testing pipeline reproducible.

```bash
dvc repro
```

The pipeline consists of two stages:

```text
build_vector_store
        ↓
    vector_db
        ↓
test_chatbot
        ↓
output/chat_output.txt
```

DVC also avoids unnecessarily rebuilding the vector database when the relevant dependencies and parameters have not changed.

To visualize the pipeline:

```bash
dvc dag
```

To check pipeline status:

```bash
dvc status
```

---

## 🧪 Experiment Tracking with DVCLive

DVCLive is used to record chatbot experiment parameters and metrics.

Tracked parameters include:

* Temperature
* Retrieval `k`
* Number of displayed chunks

Experiments can be executed with:

```bash
dvc exp run
```

and compared using:

```bash
dvc exp show
```

This makes it possible to experiment with different RAG configurations without manually rebuilding unchanged pipeline stages.

---

## 💡 How It Works

1. Upload one or more research papers.
2. Extract text from the PDF documents.
3. Split the extracted text into chunks.
4. Generate embeddings using Sentence Transformers.
5. Store the embeddings in FAISS.
6. Retrieve the most relevant document sections for a question.
7. Filter low-quality retrieved chunks for source display.
8. Pass the retrieved information to the Mistral LLM.
9. Generate an answer based strictly on the retrieved document content.
10. Display the answer along with relevant source pages.

---

## 🔐 Environment Variables

The Mistral API key is loaded through environment variables and should **never be committed to Git**.

```env
MISTRAL_API_KEY=your_api_key
```

The `.env` file is excluded through `.gitignore`.

---

## 🔮 Future Improvements

* Support additional document formats
* Improved citation highlighting
* Multi-paper comparison
* Persistent user document storage
* Conversation memory
* Better retrieval evaluation
* Retrieval and response latency tracking
* MLflow integration
* Containerized deployment
* Cloud-based DVC remote storage

---

## 👨‍💻 Project Goal

PaperMind was built to demonstrate a complete **RAG + MLOps workflow**, combining document intelligence with reproducible data processing, configurable experimentation, logging, and version-controlled pipelines.
