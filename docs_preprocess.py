from logger import get_logger
logger = get_logger("preprocess")

def clean_text(text):
    """Clean the text by removing newlines, tabs, and extra spaces."""
    try:
        logger.debug("Cleaning text")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        while "  " in text:
            text = text.replace("  ", " ")

        cleaned_text = text.strip()
        logger.info("Text cleaned successfully.")
        return cleaned_text
    
    except Exception as e:
        logger.exception("Error cleaning text: %s", str(e))
        raise

def is_low_quality_chunk(text):
    """Determine if a text chunk is of low quality based on word count and content."""
    try:
        logger.debug("Checking if chunk is low quality")
        words = text.split()

        if len(words) < 25:
            logger.info("Chunk identified as low quality due to insufficient word count.")
            return True

        if "references" in text.lower():
            logger.info("Chunk identified as low quality due to presence of 'references'.")
            return True

        logger.info("Chunk is of acceptable quality.")
        return False
    
    except Exception as e:
        logger.exception("Error checking chunk quality: %s", str(e))
        raise

def format_docs(docs : list) -> str:
    """Format documents by cleaning their text content."""
    try:
        logger.debug("Formatting documents")
        formatted_docs = "\n\n".join(clean_text(doc.page_content) for doc in docs)
        logger.info("Documents formatted successfully.")
        return formatted_docs
    
    except Exception as e:
        logger.exception("Error formatting documents: %s", str(e))
        raise
