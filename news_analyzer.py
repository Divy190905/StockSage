
import requests
from collections import Counter
from groq import Groq

# Initialize the Groq API client
client = Groq(api_key="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")

def fetch_news(stock_name=None):
    news_api_key = '3a75f8781761465da09403ccbe14a19f'
    news_base_url = 'https://newsapi.org/v2/everything'
    params = {
        'apiKey': news_api_key,
        'language': 'en',
        'pageSize': 5,
        'sortBy': 'publishedAt'
    }
    params['q'] = stock_name or 'Indian Stock Market OR Global Market Factors'
    try:
        response = requests.get(news_base_url, params=params)
        response.raise_for_status()
        news_data = response.json()
        if news_data['totalResults'] == 0 or len(news_data['articles']) < 5:
            params['q'] = 'Indian Stock Market OR Global Market Factors'
            response = requests.get(news_base_url, params=params)
            response.raise_for_status()
            news_data = response.json()
            
        articles = []
        for article in news_data['articles']:
            articles.append({
                'title': article['title'],
                'description': article['description'] or "No description available",
                'url': article['url'],
                'source': article['source'].get('name', 'Unknown'),
                'published': article['publishedAt']
            })
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

def recommend_top_stocks_from_article(article_content):
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "user",
                "content": (
                    f"Based on the following news article, list the top 5 Indian stock names "
                    f"mentioned in the article. Only return the stock names, one per line. "
                    f"If no stock news is present, recommend general stocks:\n\n{article_content}"
                )
            }],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        recommendation = ""
        for chunk in completion:
            recommendation += chunk.choices[0].delta.content or ""
        stocks = [stock.strip() for stock in recommendation.strip().split('\n') if stock.strip()]
        return stocks[:5]
    except Exception as e:
        print(f"Error fetching recommendations from GROQ API: {e}")
        return []

def process_news_articles(news_articles):
    all_stocks = []
    stocks_by_article = {}
    
    for i, article in enumerate(news_articles):
        article_content = f"{article['title']}\n{article['description']}"
        recommended_stocks = recommend_top_stocks_from_article(article_content)
        
        if not recommended_stocks:
            recommended_stocks = recommend_top_stocks_from_article("General stock recommendations for the Indian market.")
        
        if len(recommended_stocks) < 2:
            recommended_stocks = list(set(recommended_stocks + ['HDFC', 'TCS', 'Nifty']))
        
        all_stocks.extend(recommended_stocks)
        stocks_by_article[i] = {
            'article': article,
            'recommended_stocks': recommended_stocks
        }
    
    counter = Counter(all_stocks)
    top_stocks = [stock for stock, _ in counter.most_common(10)]
    
    return {
        'top_stocks': top_stocks,
        'stocks_by_article': stocks_by_article
    }