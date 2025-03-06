import gradio as gr
from stock_data import get_stock_data, predict_stock_prices
from news_analyzer import fetch_news, process_news_articles
from reddit_analyzer import analyze_reddit_sentiment
import pandas as pd
import traceback
import random

# Add this function to create a pulsing welcome animation
def create_welcome_screen():
    # Color scheme
    primary_color = "#1E88E5"
    secondary_color = "#43A047"
    dark_color = "#212121"
    
    # List of inspiring stock market quotes
    quotes = [
        "The stock market is a device for transferring money from the impatient to the patient. - Warren Buffett",
        "In the short run, the market is a voting machine. In the long run, it is a weighing machine. - Benjamin Graham",
        "The best investment you can make is an investment in yourself. - Warren Buffett",
        "Risk comes from not knowing what you're doing. - Warren Buffett",
        "The four most dangerous words in investing are: 'This time it's different.' - Sir John Templeton"
    ]
    
    # Create a visually appealing welcome screen with CSS animations
    welcome_html = f"""
    <div style="text-align:center; padding:30px; animation: fadeIn 1.5s ease-in-out;">
        <h1 style="font-size:48px; margin-bottom:5px; color:{primary_color}; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
            <span style="color:{dark_color};">Stock</span>Sage
        </h1>
        <p style="font-size:18px; margin-top:0; margin-bottom:30px; color:#555;">Intelligent Stock Analysis & Prediction</p>
        
        <div style="max-width:600px; margin:0 auto; padding:20px; border-radius:10px; background:white; box-shadow:0 4px 15px rgba(0,0,0,0.1);">
            <div style="text-align:center; margin-bottom:20px;">
                <div style="font-size:60px; animation: pulse 2s infinite;">📈</div>
            </div>
            
            <p style="font-style:italic; color:#555; padding:15px; border-left:4px solid {secondary_color}; background:#f9f9f9;">
                "{random.choice(quotes)}"
            </p>
            
            <p style="margin-top:25px;">Enter a stock ticker above to begin your analysis!</p>
        </div>
        
        <div style="margin-top:30px; font-size:14px; color:#777;">
            <p>Powered by LSTM Networks, Natural Language Processing & Sentiment Analysis</p>
        </div>
    </div>
    
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.1); }}
        100% {{ transform: scale(1); }}
    }}
    </style>
    """
    
    return welcome_html

def display_stock_data(ticker):
    if not ticker:
        return "Please enter a stock ticker symbol", None
    
    data = get_stock_data(ticker)
    if "Error" in data:
        return f"Error fetching stock data: {data['Error']}", None
    
    # Extract chart from data
    chart = data.pop("chart", None)
    chart_html = f"<img src='data:image/png;base64,{chart}' width='100%'>" if chart else None
    
    # Format the data into a nice HTML table
    html = "<h2>Financial Data</h2>"
    html += "<table style='width:100%; border-collapse: collapse;'>"
    html += "<tr><th style='text-align:left; padding:8px; border:1px solid #ddd; background-color:#f2f2f2;'>Metric</th>"
    html += "<th style='text-align:left; padding:8px; border:1px solid #ddd; background-color:#f2f2f2;'>Value</th></tr>"
    
    for key, value in data.items():
        # Format values (add commas to large numbers, format percentages, etc.)
        if isinstance(value, (int, float)) and key != "Day High" and key != "Day Low":
            if "Ratio" in key:
                formatted_value = f"{value:.2f}" if value != "N/A" else "N/A"
            elif key == "Market Cap":
                formatted_value = f"${value:,}" if value != "N/A" else "N/A"
            elif "Growth" in key or "Equity" in key:
                formatted_value = f"{value*100:.2f}%" if value != "N/A" else "N/A"
            else:
                formatted_value = f"${value:.2f}" if value != "N/A" else "N/A"
        else:
            formatted_value = value
            
        html += f"<tr><td style='text-align:left; padding:8px; border:1px solid #ddd;'>{key}</td>"
        html += f"<td style='text-align:left; padding:8px; border:1px solid #ddd;'>{formatted_value}</td></tr>"
    
    html += "</table>"
    
    return html, chart_html

def analyze_news(ticker):
    articles = fetch_news(ticker if ticker else None)
    if not articles:
        return "No news articles found", ""
    
    result = process_news_articles(articles)
    top_stocks = result['top_stocks']
    stocks_by_article = result['stocks_by_article']
    
    # Format top stocks as HTML
    top_stocks_html = "<h2>Top Recommended Stocks From News</h2>"
    top_stocks_html += "<ul>"
    for stock in top_stocks:
        top_stocks_html += f"<li>{stock}</li>"
    top_stocks_html += "</ul>"
    
    # Format articles and their recommendations
    articles_html = "<h2>News Articles with Stock Recommendations</h2>"
    for idx, item in stocks_by_article.items():
        article = item['article']
        stocks = item['recommended_stocks']
        
        articles_html += f"<div style='margin-bottom:20px; padding:10px; border:1px solid #ddd; border-radius:5px;'>"
        articles_html += f"<h3>{article['title']}</h3>"
        articles_html += f"<p><strong>Source:</strong> {article['source']} | <strong>Published:</strong> {article['published']}</p>"
        articles_html += f"<p>{article['description']}</p>"
        articles_html += f"<p><a href='{article['url']}' target='_blank'>Read more</a></p>"
        
        articles_html += "<p><strong>Recommended Stocks:</strong></p><ul>"
        for stock in stocks:
            articles_html += f"<li>{stock}</li>"
        articles_html += "</ul></div>"
    
    return top_stocks_html, articles_html

def analyze_reddit(ticker):
    try:
        top_stocks, posts = analyze_reddit_sentiment(ticker if ticker else None)
        
        if not posts:
            return "No Reddit posts found. There might be an issue with the Reddit API connection.", ""
        
        # Format top stocks as HTML
        top_stocks_html = "<h2>Top Recommended Stocks From Reddit</h2>"
        top_stocks_html += "<ul>"
        for stock in top_stocks:
            top_stocks_html += f"<li>{stock}</li>"
        top_stocks_html += "</ul>"
        
        # Format posts and their recommendations
        posts_html = "<h2>Reddit Posts with Stock Recommendations</h2>"
        for post in posts:
            sentiment_color = "#4CAF50" if post['sentiment'] == "Positive" else "#f44336" if post['sentiment'] == "Negative" else "#9E9E9E"
            
            posts_html += f"<div style='margin-bottom:20px; padding:10px; border:1px solid #ddd; border-radius:5px;'>"
            posts_html += f"<h3>{post['title']}</h3>"
            posts_html += f"<p><strong>Sector:</strong> {post['sector']} | "
            posts_html += f"<strong>Sentiment:</strong> <span style='color:{sentiment_color};'>{post['sentiment']}</span> | "
            posts_html += f"<strong>Upvotes:</strong> {post['upvotes']}</p>"
            posts_html += f"<p><a href='{post['url']}' target='_blank'>View on Reddit</a></p>"
            
            posts_html += "<p><strong>Recommended Stocks:</strong></p><ul>"
            for stock in post.get('recommended_stocks', []):
                posts_html += f"<li>{stock}</li>"
            posts_html += "</ul></div>"
        
        return top_stocks_html, posts_html
    except Exception as e:
        error_msg = f"Error analyzing Reddit data: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg, ""

def predict_prices(ticker):
    try:
        if not ticker:
            return "Please enter a stock ticker symbol", None
        
        metrics, forecast_plot = predict_stock_prices(ticker)
        
        if forecast_plot:
            forecast_img = f"<img src='data:image/png;base64,{forecast_plot}' width='100%'>"
            return metrics, forecast_img
        else:
            return metrics, None
    except Exception as e:
        error_msg = f"Error during stock price prediction: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg, None

# Create Gradio interface
with gr.Blocks(title="StockSage") as app:
    gr.Markdown("# StockSage Dashboard")
    
    with gr.Row():
        ticker_input = gr.Textbox(label="Stock Ticker Symbol (e.g., AAPL for Apple, TATAMOTORS.NS for Tata Motors)", placeholder="Enter stock ticker...")
    
    # Display welcome screen initially
    welcome_html = gr.HTML(create_welcome_screen())
    
    with gr.Tabs():
        with gr.TabItem("Financial Data"):
            analyze_btn = gr.Button("Get Financial Data")
            financial_output = gr.HTML()
            stock_chart = gr.HTML()
            analyze_btn.click(fn=display_stock_data, inputs=ticker_input, outputs=[financial_output, stock_chart])
            # Hide welcome screen when button is clicked
            analyze_btn.click(fn=lambda: "", outputs=welcome_html)
        
        with gr.TabItem("News Analysis"):
            news_btn = gr.Button("Analyze News")
            with gr.Row():
                top_stocks_news = gr.HTML()
                news_details = gr.HTML()
            news_btn.click(fn=analyze_news, inputs=ticker_input, outputs=[top_stocks_news, news_details])
        
        with gr.TabItem("Reddit Sentiment"):
            reddit_btn = gr.Button("Analyze Reddit Sentiment")
            with gr.Row():
                top_stocks_reddit = gr.HTML()
                reddit_details = gr.HTML()
            reddit_btn.click(fn=analyze_reddit, inputs=ticker_input, outputs=[top_stocks_reddit, reddit_details])
        
        with gr.TabItem("Price Prediction"):
            predict_btn = gr.Button("Predict Stock Prices")
            metrics_output = gr.Textbox(label="Model Metrics")
            forecast_chart = gr.HTML()
            
            predict_btn.click(
                fn=predict_prices, 
                inputs=ticker_input, 
                outputs=[metrics_output, forecast_chart]
            )
            # Hide welcome screen when button is clicked
            predict_btn.click(fn=lambda: "", outputs=welcome_html)

# Launch the app
if __name__ == "__main__":
    import os
    # Print debug information
    print(f"Working directory: {os.getcwd()}")
    print("Starting Stock Analyst dashboard...")
    app.launch()