#!/usr/bin/env python3
"""
Data Formatter / Cleaner for Sentiment Analysis Project
This script performs basic cleaning on raw review data:
- Removes duplicates
- Handles missing values
- Cleans text (lowercase, remove HTML tags, punctuation, emojis, etc.)
- Optional: removes stopwords and performs lemmatization
Usage:
    python src/formatter.py --input data/raw_reviews.csv --output data/cleaned_reviews.csv
"""
import argparse
import re
import pandas as pd
# import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Download NLTK resources (run once)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove emojis and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Optional: Remove stopwords and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
def main():
    parser = argparse.ArgumentParser(description="Clean raw review data for sentiment analysis")
    parser.add_argument('--input', type=str, required=True, help='Path to input raw CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output cleaned CSV file')
    args = parser.parse_args()
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Original shape: {df.shape}")
    # Basic data cleaning
    df.drop_duplicates(inplace=True)
    # Assume the main text column is named 'review' or 'text' or 'content'
    # Change the column name below if your raw file uses a different name
    text_column = None
    possible_columns = ['review', 'text', 'content', 'verified_reviews', 'comment', 'body']
    for col in possible_columns:
        if col in df.columns:
            text_column = col
            break
    if text_column is None:
        raise ValueError(f"Could not find text column. Available columns: {list(df.columns)}")
    print(f"Using text column: '{text_column}'")
    # Handle missing values in text column
    df = df.dropna(subset=[text_column])
    # Clean the text
    print("Cleaning text data...")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    # Drop rows where cleaned text became empty
    df = df[df['cleaned_text'].str.strip() != '']
    print(f"Cleaned shape: {df.shape}")
    # Save the cleaned dataset
    df.to_csv(args.output, index=False)
    print(f"Cleaned data saved to {args.output}")
if __name__ == "__main__":
    main()