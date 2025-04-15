# 10-k Filings Analysis

SigmaNLP is a Streamlit-based application designed for performing sentiment analysis on 10-K filings of publicly traded companies. Using advanced natural language processing (NLP) techniques, including FinBERT, SigmaNLP analyzes various sections of 10-K filings to compute sentiment ratings and visualize trends over time. Additionally, it fetches related news articles using NewsAPI to provide additional context for the analyzed company.

## Features

- **Sentiment Analysis**:
  - Analyze specific sections of 10-K filings (e.g., "Business Overview", "Risk Factors").
  - Uses FinBERT, a pre-trained sentiment analysis model for financial text.
- **Visualization**:
  - Displays overall sentiment trends over the years using a line graph.
  - Provides a color-coded table for easy interpretation of sentiment scores.
- **News Integration**:
  - Fetches and displays related news articles for the company ticker using NewsAPI.

## Installation

To run SigmaNLP locally, follow these steps:

### Prerequisites

1. Python 3.8 or higher.
2. [Git](https://git-scm.com/).
3. [Streamlit](https://streamlit.io/).
4. [NewsAPI Key](https://newsapi.org/) (required for fetching related news articles).
5. [TextRazor API Key](https://www.textrazor.com/) (optional, for extended text analysis).

### Clone the Repository

```bash
git clone https://github.com/varchan123/sigmanlp.git
cd sigmanlp
```

### Set Up a Virtual Environment

It is recommended to use a virtual environment for better dependency management:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Install the required Python dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Export API Keys

Ensure you export your NewsAPI and TextRazor API keys as environment variables:

```bash
export NEWSAPI_KEY=your_newsapi_key
export TEXTRAZOR_KEY=your_textrazor_key
```

Replace `your_newsapi_key` and `your_textrazor_key` with your actual API keys.

## Usage

To run the application locally, execute the following command:

```bash
streamlit run streamlit_app.py
```

Once the application is running, visit [http://localhost:8501](http://localhost:8501) in your web browser to access the interface.

## How It Works

1. **Input**: 
   - Enter the stock ticker (e.g., GOOG for Google) and the start year (e.g., 2019) in the input fields.
   
2. **Processing**:
   - The application downloads 10-K filings using an Edgar crawler.
   - Sentiment analysis is performed on specific sections of the filings using FinBERT.

3. **Output**:
   - A line graph shows the overall sentiment trend over the years.
   - A table displays sentiment scores for each section, color-coded for better clarity.
   - Related news articles are listed at the bottom of the interface.

## File Structure

- `streamlit_app.py`: Main Streamlit application code.
- `requirements.txt`: Python dependencies.
- `edgar-crawler/`: Contains scripts for downloading and processing 10-K filings.
- `datasets/`: Stores raw and extracted filings data.

## Example Workflow

1. Enter the stock ticker (e.g., `AAPL`) and the start year (e.g., `2020`).
2. Click the **Analyze** button.
3. The application will:
   - Download and process the 10-K filings.
   - Perform sentiment analysis.
   - Display the results as a graph and table.
   - Fetch related news articles for the specified company.

## Key Technologies

1. **Streamlit**: Provides the interactive web-based user interface.
2. **FinBERT**: A transformer-based model fine-tuned for sentiment analysis on financial text.
3. **NewsAPI**: Fetches related news articles to provide additional insights.
4. **TextRazor** (Optional): Advanced text analysis API for extracting topics, entities, and summaries.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes or new features.

## Support

For issues or questions, please open an issue in the [GitHub repository](https://github.com/varchan123/sigmanlp/issues).
