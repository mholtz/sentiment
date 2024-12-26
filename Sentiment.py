import praw
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from datetime import datetime
import configparser
import argparse

# Download required NLTK data
nltk.download('vader_lexicon')

def setup_reddit_client():
    """
    Set up and return Reddit API client using credentials from config.ini
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    return praw.Reddit(
        client_id=config['REDDIT']['client_id'],
        client_secret=config['REDDIT']['client_secret'],
        user_agent=config['REDDIT']['user_agent']
    )

def analyze_sentiment(text):
    """
    Analyze sentiment of given text using VADER sentiment analyzer
    Returns compound polarity score (-1 to 1)
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def get_article_data(reddit, urls):
    """
    Retrieve and analyze sentiment for given Reddit article URLs
    """
    results = []
    
    for url in urls:
        try:
            # Extract submission ID from URL
            submission_id = reddit.submission(url=url)
            
            # Get submission data
            title = submission_id.title
            body = submission_id.selftext
            comments = submission_id.comments
            
            # Combine title and body for overall article sentiment
            article_text = f"{title} {body}"
            article_sentiment = analyze_sentiment(article_text)
            
            # Analyze top-level comments
            comment_sentiments = []
            comments.replace_more(limit=0)  # Only get readily available comments
            
            for comment in comments.list()[:20]:  # Analyze top 20 comments
                comment_sentiments.append(analyze_sentiment(comment.body))
            
            avg_comment_sentiment = sum(comment_sentiments) / len(comment_sentiments) if comment_sentiments else 0
            
            results.append({
                'url': url,
                'title': title,
                'article_sentiment': article_sentiment,
                'comment_sentiment': avg_comment_sentiment,
                'overall_sentiment': (article_sentiment + avg_comment_sentiment) / 2,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze sentiment of Reddit articles')
    parser.add_argument('urls', nargs='+', help='Reddit article URLs to analyze')
    parser.add_argument('--output', '-o', help='Output CSV file name', default='sentiment_analysis.csv')
    
    args = parser.parse_args()
    
    # Initialize Reddit client
    reddit = setup_reddit_client()
    
    # Analyze articles
    results_df = get_article_data(reddit, args.urls)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    
    # Print results
    print("\nSentiment Analysis Results:")
    print("==========================")
    for _, row in results_df.iterrows():
        print(f"\nArticle: {row['title']}")
        print(f"URL: {row['url']}")
        print(f"Article Sentiment: {row['article_sentiment']:.3f}")
        print(f"Average Comment Sentiment: {row['comment_sentiment']:.3f}")
        print(f"Overall Sentiment: {row['overall_sentiment']:.3f}")

if __name__ == "__main__":
    main()