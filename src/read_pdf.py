import warnings
warnings.filterwarnings("ignore")
import os
import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.logger import get_logger

logger = get_logger("read_pdf")

def load_params_from_yaml(file_path : str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
            logger.info("Parameters loaded successfully from %s.", file_path)
            return params["read_pdf"]
    except FileNotFoundError:
        logger.error("YAML file not found: %s", file_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file: %s", str(e))
        raise

def load_documents_from_pdfs(pdf_folder : str) -> list:
    """Load documents from PDF files in the specified folder."""
    try:
        all_documents = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                path = os.path.join(pdf_folder, filename)
                loader = PyPDFLoader(path)
                docs = loader.load()
                all_documents.extend(docs)
                logger.info("Loaded %s successfully having %d pages.", filename, len(docs))

        if not all_documents:
            logger.error("No PDF documents found in the data folder.")
            raise ValueError("No PDF documents found in the data folder.")

        logger.info("Total documents loaded: %d", len(all_documents))
        return all_documents

    except Exception as e:
        logger.exception("Error loading documents from PDFs: %s", str(e))
        raise


def split_documents_into_chunks(documents : list, params : dict) -> list:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.error("No chunks were created from the uploaded PDFs.")
            raise ValueError("No chunks were created from the uploaded PDFs.")

        logger.info("Chunking done: %d", len(chunks))
        return chunks

    except Exception as e:
        logger.exception("Error splitting documents into chunks: %s", str(e))
        raise

def load_embedding_model() -> HuggingFaceEmbeddings:
    """Create an embedding model using HuggingFaceEmbeddings."""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
        return embedding_model

    except Exception as e:
        logger.exception("Error while loading embedding model: %s", str(e))
        raise

def create_vectorstore(chunks : list, embedding_model : HuggingFaceEmbeddings) -> FAISS:
    """Create a FAISS vector store from document chunks."""
    try:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )
        logger.info("Vector store created successfully with %d vectors.", len(vectorstore.index_to_docstore_id))
        return vectorstore

    except Exception as e:
        logger.exception("Error creating vector store: %s", str(e))
        raise

def store_vectorstore(vectorstore : FAISS, path : str = "vector_db"):
    """Store the FAISS vector store locally."""
    try:
        vectorstore.save_local(path)
        logger.info("Vector database saved successfully at %s.", path)

    except Exception as e:
        logger.exception("Error saving vector store: %s", str(e))
        raise

def build_vector_store():
    try:
        params = load_params_from_yaml("params.yaml")
        pdf_folder = "data"
        documents = load_documents_from_pdfs(pdf_folder)
        chunks = split_documents_into_chunks(documents, params)
        embedding_model = load_embedding_model()
        vectorstore = create_vectorstore(chunks, embedding_model)
        store_vectorstore(vectorstore)
        logger.info("Vector store creation process completed successfully.")

    except Exception as e:
        logger.error("An error occurred in the vector store creation process: %s", str(e))

if __name__ == "__main__":
    build_vector_store()