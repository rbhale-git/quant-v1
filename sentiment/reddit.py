import praw
from config import settings


class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )

    def fetch_posts(self, subreddits: list[str] = None, limit: int = 100) -> list[dict]:
        subreddits = subreddits or settings.reddit_subreddits
        posts = []
        for sub_name in subreddits:
            subreddit = self.reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=limit):
                posts.append({
                    "title": post.title,
                    "text": f"{post.title} {post.selftext}",
                    "upvotes": post.score,
                    "subreddit": sub_name,
                    "created_utc": post.created_utc,
                    "num_comments": post.num_comments,
                })
                # Also grab top comments
                post.comments.replace_more(limit=0)
                for comment in post.comments[:5]:
                    posts.append({
                        "text": comment.body,
                        "upvotes": comment.score,
                        "subreddit": sub_name,
                        "created_utc": comment.created_utc,
                    })
        return posts

    def fetch_posts_for_ticker(self, ticker: str, subreddits: list[str] = None, limit: int = 50) -> list[dict]:
        subreddits = subreddits or settings.reddit_subreddits
        posts = []
        for sub_name in subreddits:
            subreddit = self.reddit.subreddit(sub_name)
            for post in subreddit.search(ticker, limit=limit, time_filter="day"):
                posts.append({
                    "text": f"{post.title} {post.selftext}",
                    "upvotes": post.score,
                    "ticker": ticker,
                    "subreddit": sub_name,
                    "created_utc": post.created_utc,
                })
        return posts
