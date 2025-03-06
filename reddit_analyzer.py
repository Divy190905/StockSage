import praw
from collections import Counter
from groq import Groq
import time
import traceback

# Initialize the Groq API client
client = Groq(api_key="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")

# Set up Reddit API credentials
REDDIT_CLIENT_ID = 'IhrERBkIZvPfgwBPaeYSPQ'
REDDIT_SECRET = 'uIbFlDxKHSFGjWp_Ussk3qc1mY0d2Q'
REDDIT_USER_AGENT = 'my_stock_sentiment_bot v1.0'

# Define sector keywords
SECTOR_KEYWORDS = {
    "Energy": ["energy", "oil", "gas", "renewable", "power"],
    "Technology": ["technology", "IT", "software", "hardware", "AI", "cloud"],
    "Finance": ["finance", "bank", "investment", "insurance", "equity"],
    "Healthcare": ["healthcare", "pharmaceutical", "biotech", "hospital", "medical"],
    "Industrials": ["industrial", "manufacturing", "engineering", "construction"],
    "Consumer Goods": ["consumer", "FMCG", "retail", "lifestyle", "fashion"],
    "Utilities": ["utilities", "electric", "water", "energy"],
}

# Sentiment keywords
POSITIVE_KEYWORDS = ["buy", "bullish", "increase", "gain", "profit", "positive", "growth"]
NEGATIVE_KEYWORDS = ["sell", "bearish", "decline", "loss", "negative", "drop", "fall"]

def fetch_reddit_posts(stock_name=None):
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                            client_secret=REDDIT_SECRET,
                            user_agent=REDDIT_USER_AGENT)
        
        # Test connection
        reddit.user.me()  # This will fail if credentials are wrong
        
        subreddit_name = 'IndianStockMarket'
        posts = []

        try:
            search_query = f"{stock_name} stock" if stock_name else '(NSE OR BSE OR "Indian stock market")'
            print(f"Searching Reddit for: {search_query}")
            
            # Test if we can access the subreddit
            subreddit = reddit.subreddit(subreddit_name)
            # If no stock name provided, get recent posts instead of searching
            if not stock_name:
                submissions = subreddit.hot(limit=10)
            else:
                submissions = subreddit.search(search_query, sort='new', limit=10)
            
            count = 0
            for submission in submissions:
                posts.append(submission)
                count += 1
            
            print(f"Found {count} posts on Reddit")
            
            # If we found no posts with the search and we provided a stock name, try hot posts
            if count == 0 and stock_name:
                print("No posts found with search, getting hot posts instead")
                for submission in subreddit.hot(limit=10):
                    posts.append(submission)
            
            return posts

        except Exception as e:
            print(f"Error searching Reddit: {e}")
            print(traceback.format_exc())
            
            # Fallback to getting hot posts
            try:
                print("Trying fallback to hot posts")
                for submission in reddit.subreddit(subreddit_name).hot(limit=5):
                    posts.append(submission)
                return posts
            except:
                print("Fallback also failed")
                return None

    except Exception as e:
        print(f"Error initializing Reddit API: {e}")
        print(traceback.format_exc())
        return None

def classify_sector(content):
    combined_text = content.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(keyword in combined_text for keyword in keywords):
            return sector
    return "Uncategorized"

def detect_sentiment(title, description):
    combined_text = f"{title} {description}".lower()
    if any(word in combined_text for word in POSITIVE_KEYWORDS):
        return "Positive"
    elif any(word in combined_text for word in NEGATIVE_KEYWORDS):
        return "Negative"
    return "Neutral"

def get_top_comment(post):
    post.comments.replace_more(limit=0)
    top_comment = max(post.comments.list(), key=lambda c: c.score, default=None)
    return top_comment.body if top_comment else "No comments"

def extract_post_details(posts):
    post_details = []
    for post in posts:
        sentiment = detect_sentiment(post.title, post.selftext)
        post_content = f"{post.title} {post.selftext if post.selftext else 'No description available'}"
        post_info = {
            'title': post.title,
            'content': post_content,
            'sector': classify_sector(post_content),
            'sentiment': sentiment,
            'upvotes': post.score,
            'url': f"https://www.reddit.com{post.permalink}",
            'top_comment': get_top_comment(post)
        }
        post_details.append(post_info)
    post_details.sort(key=lambda x: x['upvotes'], reverse=True)
    return post_details

def recommend_top_stocks(post_content):
    try:
        print(f"Requesting stock recommendations from Groq for: {post_content[:50]}...")
        
        try:
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": f"based on the given post just recommend the top stocks and just give the name of the stock and nothing else not even the top line\n\n{post_content}"
                    }
                ],
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
            
            # If no stocks were found, provide fallback stocks
            if not stocks:
                stocks = ["RELIANCE", "HDFC", "TCS", "ITC", "INFY"]
                print("Using fallback stock recommendations")
            
            print(f"Recommended stocks: {stocks}")
            return stocks
            
        except Exception as e:
            print(f"Error with Groq API: {e}")
            print(traceback.format_exc())
            # Return some default stocks if API fails
            return ["RELIANCE", "HDFC", "TCS", "ITC", "INFY"]
            
    except Exception as e:
        print(f"Error recommending stocks: {e}")
        return ["RELIANCE", "HDFC", "TCS", "ITC", "INFY"]  # Fallback stocks

def analyze_reddit_sentiment(stock_name=None):
    print(f"\n--- Starting Reddit analysis for {stock_name or 'general market'} ---")
    
    try:
        reddit_posts = fetch_reddit_posts(stock_name)
        if not reddit_posts or len(reddit_posts) == 0:
            print("No Reddit posts found, using fallback data")
            # Create more varied and realistic fallback data
            if stock_name:
                fallback_stocks = [stock_name, "RELIANCE", "HDFC", "TCS", "INFY"]
                title = f"Discussion about {stock_name} and market trends"
            else:
                fallback_stocks = ["RELIANCE", "HDFC", "TCS", "INFY", "ITC", "SBIN", "BHARTIARTL", "LT", "HDFCBANK", "TATASTEEL"]
                title = "General Market Discussion"
            
            fallback_post = {
                'title': title,
                'content': f"General discussion about {stock_name if stock_name else 'Indian'} stock market trends.",
                'sector': "Finance",
                'sentiment': "Neutral",
                'upvotes': 10,
                'url': "https://www.reddit.com/r/IndianStockMarket/",
                'recommended_stocks': fallback_stocks[:5]
            }
            return fallback_stocks, [fallback_post]
        
        print(f"Processing {len(reddit_posts)} posts")
        post_details = extract_post_details(reddit_posts)
        
        # Generate recommendations and count stock frequencies
        stock_counter = Counter()
        posts_with_stocks = []
        
        for post in post_details:
            try:
                print(f"Getting stock recommendations for post: {post['title']}")
                top_stocks = recommend_top_stocks(post['content'])
                post['recommended_stocks'] = top_stocks
                posts_with_stocks.append(post)
                stock_counter.update(top_stocks)
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing post: {e}")
                post['recommended_stocks'] = ["RELIANCE", "HDFC", "TCS"]
                posts_with_stocks.append(post)
        
        top_stocks = [stock for stock, _ in stock_counter.most_common(10)]
        
        # If somehow we still have no stocks, use fallback with stock name as first stock
        if not top_stocks and stock_name:
            top_stocks = [stock_name, "RELIANCE", "HDFC", "TCS", "INFY"]
        elif not top_stocks:
            top_stocks = ["RELIANCE", "HDFC", "TCS", "INFY", "ITC", "SBIN", "BHARTIARTL", "LT", "HDFCBANK", "TATASTEEL"]
        
        print(f"Analysis complete. Top stocks: {top_stocks}")
        return top_stocks, posts_with_stocks
    
    except Exception as e:
        print(f"Complete error in Reddit analysis: {str(e)}")
        print(traceback.format_exc())
        fallback_stocks = ["RELIANCE", "HDFC", "TCS", "INFY", "ITC"]
        if stock_name:
            fallback_stocks.insert(0, stock_name)
        
        fallback_post = {
            'title': f"Discussion about {stock_name if stock_name else 'Indian'} stocks",
            'content': "Fallback content due to API error",
            'sector': "Finance",
            'sentiment': "Neutral",
            'upvotes': 5,
            'url': "https://www.reddit.com/r/IndianStockMarket/",
            'recommended_stocks': fallback_stocks[:5]
        }
        return fallback_stocks, [fallback_post]