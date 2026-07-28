import warnings
warnings.filterwarnings("ignore")

import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store():

    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    pdf_folder = "data"
    all_document = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_document.extend(docs)
            print(f"Loaded {filename} successfully having {len(docs)} pages.")

    if not all_document:
        raise ValueError("No PDF documents found in the data folder.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(all_document)

    if not chunks:
        raise ValueError("No chunks were created from the uploaded PDFs.")

    print(f"Chunking done: {len(chunks)}")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    vectorstore.save_local("vector_db")
    print("Vector database created successfully.")


if __name__ == "__main__":
    build_vector_store()