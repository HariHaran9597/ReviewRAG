import os
import requests
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from fastapi import HTTPException

# Load environment variables
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")


def fetch_reviews_from_rapidapi(product_asin: str, country: str = "US") -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Fetches reviews from the Real-Time Amazon Data API on RapidAPI.
    Tries multiple pages to gather enough reviews.
    
    Returns:
        Tuple of (reviews_list_or_None, status_message)
        - reviews_list: list of mapped review dicts or None if API failed
        - status: "success", "no_reviews", or "api_error"
    """
    if not RAPIDAPI_KEY or RAPIDAPI_KEY == "your_rapidapi_key":
        return None, "api_error"

    all_reviews = []
    
    # Try the primary country first, then fallback to US if different
    countries_to_try = [country]
    if country != "US":
        countries_to_try.append("US")

    for c in countries_to_try:
        for page in range(1, 6):  # Up to 5 pages
            try:
                headers = {
                    "X-RapidAPI-Key": RAPIDAPI_KEY,
                    "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
                }
                params = {
                    "asin": product_asin,
                    "country": c,
                    "sort_by": "TOP_REVIEWS",
                    "page": str(page),
                    "verified_purchases_only": "false",
                    "images_or_videos_only": "false"
                }
                
                response = requests.get(
                    "https://real-time-amazon-data.p.rapidapi.com/product-reviews",
                    headers=headers, params=params, timeout=15
                )
                
                if response.status_code == 403:
                    print(f"[WARN] Not subscribed to Real-Time Amazon Data API")
                    return None, "api_error"
                
                if response.status_code == 429:
                    print(f"[WARN] Rate limited on page {page}")
                    break
                    
                response.raise_for_status()
                data = response.json()
                reviews = data.get("data", {}).get("reviews", [])
                total_ratings = data.get("data", {}).get("total_ratings", 0)
                
                if not reviews:
                    if total_ratings == 0 and page == 1:
                        print(f"[INFO] Product has 0 ratings on amazon.{c} — skipping")
                    break  # No more pages
                
                for r in reviews:
                    mapped = {
                        "id": r.get("review_id", ""),
                        "rating": r.get("review_star_rating", 0),
                        "title": r.get("review_title", ""),
                        "text": r.get("review_comment", ""),
                        "is_verified": "Verified" in str(r.get("review_verified_purchase", ""))
                    }
                    all_reviews.append(mapped)
                    
                print(f"[INFO] Page {page}, Country={c}: {len(reviews)} reviews (total so far: {len(all_reviews)})")
                    
            except requests.exceptions.HTTPError:
                break
            except Exception as e:
                print(f"[WARN] Error on page {page}, country={c}: {e}")
                break
        
        if len(all_reviews) >= 10:
            break
            
    if all_reviews:
        return all_reviews, "success"
    else:
        return None, "no_reviews"


def filter_reviews_strict(raw_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    STRICT filtering (the premium version):
    - Keep ONLY Verified purchases
    - Keep ONLY 3-star or 4-star reviews
    - Keep ONLY reviews with > 20 words
    """
    filtered = []
    for review in raw_reviews:
        if not review.get("is_verified", False):
            continue
        try:
            rating_val = int(float(str(review.get("rating", 0)).split()[0]))
        except (ValueError, TypeError):
            continue
        if rating_val not in [3, 4]:
            continue
        text = review.get("text", "")
        if len(str(text).split()) <= 20:
            continue
        filtered.append({
            "id": review.get("id"),
            "rating": rating_val,
            "title": review.get("title", ""),
            "text": text.strip()
        })
    return filtered


def filter_reviews_relaxed(raw_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    RELAXED filtering — used as fallback when strict yields 0.
    Keeps all reviews with > 20 words (any star rating, any verification).
    """
    filtered = []
    for review in raw_reviews:
        text = review.get("text", "")
        if len(str(text).split()) <= 20:
            continue
        try:
            rating_val = int(float(str(review.get("rating", 0)).split()[0]))
        except (ValueError, TypeError):
            rating_val = 0
        filtered.append({
            "id": review.get("id"),
            "rating": rating_val,
            "title": review.get("title", ""),
            "text": text.strip()
        })
    return filtered


def process_product_reviews(product_asin: str, country: str = "US") -> List[Dict[str, Any]]:
    """
    Main pipeline:
    1. Try to fetch reviews from RapidAPI (multiple pages + countries).
    2. Apply STRICT filter (3-4 star, verified, >20 words).
    3. If strict filter gives 0, apply RELAXED filter (any star, >20 words).
    4. If API returns no reviews or fails, raise a clear error — NEVER use fake demo data.
    """
    raw_reviews, status = fetch_reviews_from_rapidapi(product_asin, country=country)
    
    if status == "api_error":
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the Reviews API. Please check your RAPIDAPI_KEY in the .env file and ensure you are subscribed to 'Real-Time Amazon Data' API on RapidAPI."
        )
    
    if status == "no_reviews" or not raw_reviews:
        raise HTTPException(
            status_code=404,
            detail=f"No reviews found for ASIN '{product_asin}' on Amazon ({country}). This product may be too new, have zero reviews, or the ASIN may be incorrect. Try a popular product like iPhone 15 (B0CHX1W1XY) or Sony WH-1000XM4 (B0863TXGM3)."
        )
    
    print(f"[INFO] RapidAPI returned {len(raw_reviews)} total raw reviews for ASIN={product_asin}")
    
    # First try: strict 3-4 star verified filter
    strict = filter_reviews_strict(raw_reviews)
    if strict:
        print(f"[INFO] Strict filter passed: {len(strict)} reviews (3-4 star, verified, >20 words)")
        return strict
    
    # Second try: relaxed filter (any star, >20 words) 
    relaxed = filter_reviews_relaxed(raw_reviews)
    if relaxed:
        print(f"[INFO] Strict filter yielded 0. Using relaxed filter: {len(relaxed)} reviews (all stars, >20 words)")
        return relaxed
    
    # All reviews were too short
    raise HTTPException(
        status_code=400,
        detail=f"Found {len(raw_reviews)} reviews but none had enough content (>20 words) to analyze. Try a different product with more detailed reviews."
    )
