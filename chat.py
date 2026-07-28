import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def clean_chunk_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


def is_low_quality_chunk(text):
    words = text.split()

    if len(words) < 25:
        return True

    if "references" in text.lower():
        return True

    return False


def format_docs(docs):
    return "\n\n".join(clean_chunk_text(doc.page_content) for doc in docs)


def load_chatbot():
    if not os.path.exists("vector_db"):
        raise FileNotFoundError("Vector database not found. Build the knowledge base first.")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        "vector_db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template("""
You are a precise AI assistant that answers questions strictly using the retrieved context below. You do not use outside knowledge, prior training data, or assumptions beyond what is explicitly stated in the context.

## Rules for answering
1. Read all context chunks fully before answering. Synthesize information across chunks when needed.
2. Base every claim only on the provided context.
3. If the answer is completely available, answer directly.
4. If the answer is partially available, answer with the available information and mention what is missing.
5. If the answer is not present, reply exactly:
"I don't know based on the provided documents."
6. If different parts conflict, mention the conflict.
7. Never mention "chunks", "context", or "retrieval" in the answer.

## Formatting
- Summary → concise summary.
- Simple explanation → explain like a beginner.
- Comparison → markdown table.
- Pros, Cons, Features, Differences → bullet points or table.
- Process → numbered list.

Context:
{context}

Question:
{question}

Answer:
""")

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return vectorstore, chain


def ask_question(question, vectorstore, chain):
    scored_docs = vectorstore.similarity_search_with_score(question, k=5)

    display_chunks = []

    for doc, score in scored_docs:
        if not is_low_quality_chunk(doc.page_content):
            display_chunks.append((doc, score))

    if not display_chunks:
        display_chunks = scored_docs

    answer = chain.invoke(question)

    display_data = []
    for doc, score in display_chunks[:3]:
        display_data.append(
            {
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", 0) + 1,
                "text": clean_chunk_text(doc.page_content),
            }
        )

    return answer, display_data