import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """
    Removes HTML tags, multiple spaces, and normalizes the text.
    """
    # Remove HTML tags using beautifulsoup
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    
    # Remove emojis and special characters (keeping basic punctuation)
    # This regex keeps ASCII letters, numbers, and common punctuation
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def chunk_reviews(filtered_reviews: List[Dict[str, Any]], chunk_size: int = 200, chunk_overlap: int = 30) -> List[Dict[str, Any]]:
    """
    Cleans review text and splits it into smaller chunks suitable for embedding.
    Ensures that each chunk retains metadata from its parent review.
    """
    # Note: RecursiveCharacterTextSplitter works with character count by default, 
    # but we can use a token-based or word-count-based approach. 
    # For simplicity and robust default behavior, we use standard character splitting 
    # roughly equivalent to 150-200 words (~800 characters).
    
    approx_chars_per_word = 5
    char_chunk_size = chunk_size * approx_chars_per_word
    char_chunk_overlap = chunk_overlap * approx_chars_per_word
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    
    for review in filtered_reviews:
        # Clean text
        raw_text = review.get("text", "")
        cleaned_text = clean_text(raw_text)
        
        # If after cleaning, it's too short, skip
        if len(cleaned_text.split()) < 5:
            continue
            
        # Split into chunks
        split_texts = text_splitter.split_text(cleaned_text)
        
        for idx, text_chunk in enumerate(split_texts):
            chunks.append({
                "page_content": text_chunk,
                "metadata": {
                    "review_id": review.get("id"),
                    "rating": review.get("rating"),
                    "title": review.get("title"),
                    "chunk_index": idx
                }
            })
            
    return chunks
