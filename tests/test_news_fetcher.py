import json
from datetime import datetime
from unittest.mock import Mock

from collection import news_fetcher


def test_empty_daily_cache_retries_newsapi(monkeypatch, tmp_path):
    monkeypatch.setattr(news_fetcher, "_CACHE_DIR", tmp_path)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    ticker = "MSFT"
    today = datetime.today().strftime("%Y-%m-%d")
    (tmp_path / f"{ticker}_{today}.json").write_text("[]")

    response = Mock()
    response.json.return_value = {"articles": [{"title": "Microsoft news"}]}
    request = Mock(return_value=response)
    monkeypatch.setattr(news_fetcher.requests, "get", request)

    headlines = news_fetcher.fetch_news_headlines(ticker)

    assert headlines == ["Microsoft news"]
    request.assert_called_once()
    assert json.loads((tmp_path / f"{ticker}_{today}.json").read_text()) == headlines


def test_nonempty_daily_cache_skips_newsapi(monkeypatch, tmp_path):
    monkeypatch.setattr(news_fetcher, "_CACHE_DIR", tmp_path)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    ticker = "MSFT"
    today = datetime.today().strftime("%Y-%m-%d")
    cached_headlines = ["Cached Microsoft news"]
    (tmp_path / f"{ticker}_{today}.json").write_text(json.dumps(cached_headlines))
    request = Mock()
    monkeypatch.setattr(news_fetcher.requests, "get", request)

    assert news_fetcher.fetch_news_headlines(ticker) == cached_headlines
    request.assert_not_called()
