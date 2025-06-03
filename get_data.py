import json
import re
import time
from typing import Dict, List, Optional, Union

import nltk
import numpy as np
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def fetch_steam_reviews(
    app_id: Union[int, str],
    max_reviews: int = 1000,
    review_type: str = "all",
    language: str = "english",
    purchase_type: str = "all",
    delay: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch Steam reviews for a given game and return as a pandas DataFrame.
    """
    base_url = "https://store.steampowered.com/appreviews/"
    reviews_data = []
    cursor = "*"
    fetched_count = 0

    # Map review type to API parameter
    review_type_map = {"positive": "positive", "negative": "negative", "all": "all"}

    if verbose:
        print(f"Fetching Steam reviews for App ID: {app_id}")
        print(
            f"Target: {max_reviews} reviews, Type: {review_type}, Language: {language}"
        )
        print("-" * 50)

    while fetched_count < max_reviews and cursor:
        # Build API request parameters
        params = {
            "json": 1,
            "filter": review_type_map.get(review_type, "all"),
            "language": language,
            "day_range": 9223372036854775807,  # All time
            "cursor": cursor,
            "review_type": review_type_map.get(review_type, "all"),
            "purchase_type": purchase_type,
            "num_per_page": min(100, max_reviews - fetched_count),
        }

        try:
            response = requests.get(f"{base_url}{app_id}", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                if verbose:
                    print(f"API request failed: {data.get('error', 'Unknown error')}")
                break

            reviews = data.get("reviews", [])
            if not reviews:
                if verbose:
                    print("No more reviews available")
                break

            for review in reviews:
                author = review.get("author", {})
                review_data = {
                    "review_id": review.get("recommendationid"),
                    "author_steamid": author.get("steamid"),
                    "author_playtime_forever": author.get("playtime_forever", 0),
                    "review": review.get("review", "").strip(),
                    "voted_up": review.get("voted_up", False),
                    "votes_up": review.get("votes_up", 0),
                    "votes_funny": review.get("votes_funny", 0),
                }
                reviews_data.append(review_data)
                fetched_count += 1

                if fetched_count >= max_reviews:
                    break

            cursor = data.get("cursor")
            if verbose:
                print(f"Fetched {fetched_count} reviews so far...")
            time.sleep(delay)

        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            break

    if verbose:
        print(f"\nCompleted! Fetched {len(reviews_data)} reviews total")

    df = pd.DataFrame(reviews_data)
    if not df.empty:
        df["sentiment"] = df["voted_up"].map({True: "Positive", False: "Negative"})

    return df


def main():
    # PUBG: Battlegrounds App ID
    PUBG_APP_ID = "578080"

    # Fetch reviews
    print("Fetching PUBG reviews...")
    reviews_df = fetch_steam_reviews(
        app_id=PUBG_APP_ID,
        max_reviews=5000,  # Fetch 5000 reviews
        review_type="all",
        language="english",
    )

    # Save to CSV
    csv_filename = "pubg-reviews.csv"
    reviews_df.to_csv(csv_filename, index=False)
    print(f"Saved reviews to {csv_filename}")

    # Save to Parquet
    parquet_filename = "pubg-reviews.parquet"
    reviews_df.to_parquet(parquet_filename, index=False)
    print(f"Saved reviews to {parquet_filename}")

    # Print summary
    print("\nDataset Summary:")
    print(f"Total reviews: {len(reviews_df)}")
    print("\nSentiment distribution:")
    print(reviews_df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
