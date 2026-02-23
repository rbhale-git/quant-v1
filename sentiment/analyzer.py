import re
from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    # Common words that look like tickers but aren't
    TICKER_BLACKLIST = {
        "A", "I", "AM", "PM", "CEO", "CFO", "IPO", "ETF", "ATH",
        "DD", "FD", "GDP", "LLC", "USA", "USD", "SEC", "FDA",
        "IMO", "YOLO", "FOMO", "FYI", "TBH", "LOL", "OMG",
        "THE", "FOR", "AND", "BUT", "NOT", "ARE", "ALL", "HAS",
        "NOW", "NEW", "OLD", "BIG", "RUN", "PUT", "CALL",
    }

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> float:
        scores = self._vader.polarity_scores(text)
        return scores["compound"]

    def extract_tickers(self, text: str) -> list[str]:
        cashtag = re.findall(r"\$([A-Z]{1,5})\b", text)
        allcaps = re.findall(r"\b([A-Z]{2,5})\b", text)
        combined = set(cashtag + allcaps) - self.TICKER_BLACKLIST
        return sorted(combined)

    def aggregate_sentiment(self, posts: list[dict], ticker: str, as_of: date) -> dict:
        if not posts:
            return {
                "date": as_of,
                "ticker": ticker,
                "sentiment_score": 0.0,
                "mention_count": 0,
                "sentiment_trend": 0.0,
            }
        scores = []
        weights = []
        for post in posts:
            score = self.score_text(post["text"])
            weight = max(post.get("upvotes", 1), 1)
            scores.append(score)
            weights.append(weight)

        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return {
            "date": as_of,
            "ticker": ticker,
            "sentiment_score": round(weighted_score, 4),
            "mention_count": len(posts),
            "sentiment_trend": 0.0,  # Computed when historical data available
        }
