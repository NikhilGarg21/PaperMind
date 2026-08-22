import os
from dotenv import load_dotenv
import yaml
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from logger import get_logger
from docs_preprocess import format_docs , clean_text , is_low_quality_chunk
from dvclive import Live

logger = get_logger("chat")
load_dotenv()

def load_params_from_yaml(file_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
            logger.info("Parameters loaded successfully from %s.", file_path)
            return params["chat"]
    except FileNotFoundError:
        logger.error("YAML file not found: %s", file_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file: %s", str(e))
        raise

def load_embedding_model():
    """Load Embeddings model."""
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

def load_vectorstore(embedding_model : HuggingFaceEmbeddings) -> FAISS:
    """Load the FAISS vector store."""
    try:
        logger.debug("Loading vector store")
        vectorstore = FAISS.load_local(
            "vector_db",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vector store loaded successfully with %d vectors.", len(vectorstore.index_to_docstore_id))
        return vectorstore
    except FileNotFoundError as fnf_error:
        logger.error("Vector database not found: %s", str(fnf_error))
        raise
    except Exception as e:
        logger.exception("Error loading vector store: %s", str(e))
        raise

def create_retriever(vectorstore : FAISS , params : dict) -> any:
    """Create a retriever from the vector store."""
    try:
        logger.debug("Creating retriever")
        retriever = vectorstore.as_retriever(search_kwargs={"k": params["retrieval_k"]})
        logger.info("Retriever created successfully.")
        return retriever
    
    except Exception as e:
        logger.exception("Error creating retriever: %s", str(e))
        raise

def create_llm(params : dict) -> ChatMistralAI:
    """Create a ChatMistralAI model."""
    try:
        logger.debug("Creating ChatMistralAI model")
        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=params["temperature"],
        )
        logger.info("ChatMistralAI model created successfully.")
        return llm
    
    except Exception as e:
        logger.exception("Error creating ChatMistralAI model: %s", str(e))
        raise

def create_prompt_template() -> ChatPromptTemplate:
    """Create a chat prompt template."""
    try:
        logger.debug("Creating prompt template.")
        prompt = ChatPromptTemplate.from_template("""
# You are a precise AI assistant that answers questions strictly using the retrieved context below. You do not use outside knowledge, prior training data, or assumptions beyond what is explicitly stated in the context.

# ## Rules for answering
# 1. Read all context chunks fully before answering. Synthesize information across chunks when needed.
# 2. Base every claim only on the provided context.
# 3. If the answer is completely available, answer directly.
# 4. If the answer is partially available, answer with the available information and mention what is missing.
# 5. If the answer is not present, reply exactly:
# "I don't know based on the provided documents."
# 6. If different parts conflict, mention the conflict.
# 7. Never mention "chunks", "context", or "retrieval" in the answer.

# ## Formatting
# - Summary → concise summary.
# - Simple explanation → explain like a beginner.
# - Comparison → markdown table.
# - Pros, Cons, Features, Differences → bullet points or table.
# - Process → numbered list.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

        logger.info("Prompt template created successfully.")
        return prompt
    except Exception as e:  
        logger.exception("Error creating prompt template: %s", str(e))
        raise

def create_chain(retriever : any, prompt : ChatPromptTemplate, llm : ChatMistralAI) -> any:
    """Create a chain for the chatbot."""
    try:
        logger.debug("Creating chain")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Chain created successfully.")
        return chain
    except Exception as e:
        logger.exception("Error creating chain: %s", str(e))
        raise

def retrieve_docs(vectorstore : FAISS, question : str, params : dict) -> list:
    """Retrieve documents from the vector store based on the question."""
    try:
        logger.debug("Retrieving documents for the question: %s", question)
        scored_docs = vectorstore.similarity_search_with_score(question, k=params["retrieval_k"])

        display_chunks = []
        for doc, score in scored_docs:
            if not is_low_quality_chunk(doc.page_content):
                display_chunks.append((doc, score))

        if not display_chunks:
            display_chunks = scored_docs

        display_chunks = display_chunks[:params["display_chunks"]]
        logger.info("Retrieved %d relevant documents.", len(display_chunks))
        return display_chunks
    
    except Exception as e:
        logger.exception("Error retrieving documents: %s", str(e))
        raise

def ask_question(question : str, vectorstore : FAISS, chain : any, params : dict) -> tuple:
    """Ask a question and get an answer along with relevant document chunks."""
    try:
        logger.debug("Asking question: %s", question)
        display_chunks = retrieve_docs(vectorstore, question, params)
        answer = chain.invoke(question)

        display_data = []
        for doc, score in display_chunks:
            display_data.append(
                {
                    "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                    "page": doc.metadata.get("page", 0) + 1,
                    "text": clean_text(doc.page_content),
                }
            )

        logger.info("Question answered successfully.")
        return answer, display_data
    
    except Exception as e:
        logger.exception("Error asking question: %s", str(e))
        raise

def load_chatbot():
    """Main function to load the chatbot and ask a sample question."""
    try:
        params = load_params_from_yaml("params.yaml")
        embedding_model = load_embedding_model()
        vectorstore = load_vectorstore(embedding_model)
        retriever = create_retriever(vectorstore, params)
        llm = create_llm(params)
        prompt_template = create_prompt_template()
        chain = create_chain(retriever, prompt_template, llm)

        logger.info("Chatbot loaded successfully.")
        return vectorstore, chain, params

    except Exception as e:
        logger.exception("An error occurred in the main function: %s", str(e))


def main():
    """Testing function to load the chatbot and ask a sample question."""
    try:
        logger.info("Starting chatbot loading process.")
        vectorstore, chain, params = load_chatbot()

        sample_question = "What is transformer architecture?"

        with Live(save_dvc_exp=True) as live:

            live.log_param("temperature", params["temperature"])
            live.log_param("retrieval_k", params["retrieval_k"])
            live.log_param("display_chunks", params["display_chunks"])

            answer, display_data = ask_question(
                sample_question,
                vectorstore,
                chain,
                params
            )

            live.log_metric("retrieved_chunks", len(display_data))

        with open("chat_output.txt", "w", encoding="utf-8") as file:
            file.write(f"Question:\n{sample_question}\n\n")
            file.write(f"Answer:\n{answer}\n\n")
            file.write("Retrieved Chunks:\n")

            for i, chunk in enumerate(display_data, 1):
                file.write(f"\n{i}. {chunk['source']} - Page {chunk['page']}\n")
                file.write(f"{chunk['text']}\n")

        logger.info("Chat output saved successfully.")

    except Exception as e:
        logger.exception("An error occurred in the main function: %s", str(e))


if __name__ == "__main__":
    main()