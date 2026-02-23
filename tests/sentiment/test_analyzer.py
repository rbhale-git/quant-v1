import pytest
import pandas as pd
from datetime import date
from sentiment.analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


def test_score_text_positive(analyzer):
    score = analyzer.score_text("AAPL is going to the moon! Great earnings!")
    assert score > 0


def test_score_text_negative(analyzer):
    score = analyzer.score_text("This stock is crashing. Terrible outlook. Sell now.")
    assert score < 0


def test_score_text_neutral(analyzer):
    score = analyzer.score_text("The stock closed at 150.")
    assert -0.3 <= score <= 0.3


def test_aggregate_scores(analyzer):
    posts = [
        {"text": "AAPL to the moon!", "upvotes": 100, "ticker": "AAPL"},
        {"text": "AAPL great earnings beat", "upvotes": 50, "ticker": "AAPL"},
        {"text": "AAPL might drop tomorrow", "upvotes": 30, "ticker": "AAPL"},
    ]
    result = analyzer.aggregate_sentiment(posts, "AAPL", date(2024, 1, 2))
    assert result["ticker"] == "AAPL"
    assert "sentiment_score" in result
    assert "mention_count" in result
    assert result["mention_count"] == 3
    assert -1 <= result["sentiment_score"] <= 1


def test_aggregate_empty(analyzer):
    result = analyzer.aggregate_sentiment([], "AAPL", date(2024, 1, 2))
    assert result["mention_count"] == 0
    assert result["sentiment_score"] == 0.0


def test_extract_tickers(analyzer):
    text = "I'm bullish on $AAPL and TSLA. MSFT is looking good too."
    tickers = analyzer.extract_tickers(text)
    assert "AAPL" in tickers
    assert "TSLA" in tickers
    assert "MSFT" in tickers
