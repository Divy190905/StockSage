# StockSage


## Overview

StockSage is a comprehensive stock analysis platform with an intuitive web interface that combines financial data, news sentiment analysis, Reddit social sentiment, and machine learning-based price predictions to provide a 360-degree view of any stock.

## Key Features

- **Financial Data Dashboard**: Retrieve and visualize key financial metrics, historical performance, and current price data for any publicly traded stock
- **News Sentiment Analysis**: Analyze the latest news articles related to a stock and extract recommendations based on content sentiment
- **Reddit Community Sentiment**: Gauge market sentiment from Reddit discussions, classified by sector and emotional tone
- **AI-Powered Price Prediction**: Forecast future stock prices using advanced LSTM neural networks trained on 3 years of historical data
- **Beautiful Visualization**: Clean, interactive charts and data presentation with an elegant UI

## Technology Stack

- **Machine Learning**: PyTorch LSTM neural networks for time series forecasting
- **Natural Language Processing**: Groq API for sentiment analysis and stock recommendations
- **Data Sources**: Yahoo Finance API, NewsAPI, Reddit API
- **Web Interface**: Gradio interactive UI
- **Data Visualization**: Matplotlib for generating insightful charts

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Divy190905/StockSage.git
   cd StockSage
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - You'll need API keys for Groq and NewsAPI
   - Reddit API credentials if you want to use the Reddit sentiment analysis

## Usage

Run the application:
```bash
python app.py
```

Enter the stock ticker symbol in the input field:
- For US stocks: Use symbols like AAPL (Apple), MSFT (Microsoft), GOOGL (Google)
- For Indian stocks: Use NSE symbols like TATAMOTORS.NS, RELIANCE.NS, INFY.NS
- Leave blank for general market analysis

### Using the Dashboard

1. **Financial Data**: View current price, market cap, and other key financial metrics
2. **News Analysis**: See what news articles are saying about your stock and get recommendations
3. **Reddit Sentiment**: Analyze social media sentiment and identify trending stocks
4. **Price Prediction**: Get 30-day forecasts with expected price changes and risk assessment

## Example Output

The application provides:
- Stock pricing data with historical charts
- News sentiment analysis with article summaries
- Reddit discussion analysis with sentiment classification
- 30-day price forecasts with confidence metrics

## Tips for Best Results

- For Indian stocks, add the ".NS" suffix for NSE listings (e.g., "RELIANCE.NS")
- Ensure you have a stable internet connection as the app fetches real-time data
- Some less popular stocks may have limited data available for analysis
- The prediction model works best for stocks with consistent trading history



## Acknowledgments

- Data provided by Yahoo Finance, NewsAPI, and Reddit
- Sentiment analysis powered by Groq API
- Built with PyTorch and Gradio