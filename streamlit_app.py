import streamlit as st
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import subprocess

# Load FinBERT model and tokenizer
@st.cache_resource
def load_finbert():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_finbert()

def analyze_sentiment_with_rating(text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()[0]
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)

    # Assign a rating based on probabilities
    # Weighted average for the rating: [Neutral, Positive, Negative] → [3, 5, 1]
    rating_weights = [3, 5, 1]
    rating = sum(prob * weight for prob, weight in zip(probabilities, rating_weights))

    return rating, probabilities

def download_and_process_filings(ticker, start_year):
    """
    Downloads 10-K filings for the specified ticker and start year, processes the data,
    and returns a list of dictionaries containing extracted content.
    """
    # Clone the repository if not already cloned
    repo_dir = "edgar-crawler"
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/nlpaueb/edgar-crawler.git"], check=True)
    
    # Navigate to the repository directory
    os.chdir(repo_dir)

    # Create the configuration file
    from datetime import datetime
    current_year = datetime.now().year
    config = {
        "download_filings": {
            "start_year": start_year,
            "end_year": current_year,
            "quarters": [1, 2, 3, 4],
            "filing_types": ["10-K"],
            "cik_tickers": [ticker],
            "user_agent": "Your Name (your-email@example.com)",
            "raw_filings_folder": "RAW_FILINGS",
            "indices_folder": "INDICES",
            "filings_metadata_file": "FILINGS_METADATA.csv",
            "skip_present_indices": True
        },
        "extract_items": {
            "raw_filings_folder": "RAW_FILINGS",
            "extracted_filings_folder": "EXTRACTED_FILINGS",
            "filings_metadata_file": "FILINGS_METADATA.csv",
            "filing_types": ["10-K"],
            "include_signature": False,
            "items_to_extract": [],
            "remove_tables": True,
            "skip_extracted_filings": True
        }
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Run the download and extract scripts
    subprocess.run(["python", "download_filings.py"], check=True)
    subprocess.run(["python", "extract_items.py"], check=True)

    # Change back to the original directory
    os.chdir('..')

def extract_all_json_content(folder_path):
    """
    Extracts all content from JSON files in the specified folder, using only the first three parts of the filename.
    """
    extracted_content = []

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        st.error(f"Error: The folder '{folder_path}' does not exist.")
        return extracted_content

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Process only JSON files
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Extract the first three components from the filename
                parts = file_name.replace(".json", "").split("_")[:3]
                if len(parts) < 3:
                    st.warning(f"Skipping invalid filename: {file_name}")
                    continue

                cik, filing_type, year = parts

                # Load the JSON content
                with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
                    content = json.load(f)

                # Add metadata to the content
                content["cik"] = cik
                content["filing_type"] = filing_type
                content["year"] = year

                # Append the content to the list
                extracted_content.append(content)
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")

    return extracted_content

# Streamlit app UI
st.title("10-K Filings Sentiment Analysis")

# Input fields for ticker and start year
ticker = st.text_input("Enter the stock ticker (e.g., GOOG):", "GOOG")
start_year = st.number_input("Enter the start year (e.g., 2019):", min_value=2000, max_value=2025, value=2019)

if st.button("Analyze"):
    st.info("Downloading and processing filings. This may take a few minutes...")
    download_and_process_filings(ticker, start_year)

    # Extract JSON content
    folder_path = os.path.join("edgar-crawler", "datasets", "EXTRACTED_FILINGS", "10-K")
    data = extract_all_json_content(folder_path)

    if data:
        st.success(f"Successfully processed {len(data)} filings.")

        # Initialize a DataFrame for sentiment analysis
        columns = [f'item_{i}' for i in range(1, 17)]
        df = pd.DataFrame(columns=['year'] + columns)

        # Process each filing
        for report in data:
            row = {'year': report['year']}
            for item in columns:
                if item in report:
                    text = report[item]
                    rating, _ = analyze_sentiment_with_rating(text)
                    row[item] = rating
                else:
                    row[item] = None  # Handle missing items

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Display the DataFrame
        st.subheader(f"Sentiment Analysis for {ticker}")
        st.dataframe(df)

        # Save the DataFrame to CSV
        output_file = f"{ticker}_10k_sentiment_analysis.csv"
        df.to_csv(output_file, index=False)
        st.success(f"Results saved to {output_file}")
