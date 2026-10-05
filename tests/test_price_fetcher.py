import pandas as pd
from datetime import datetime, timedelta
import os
from unittest.mock import Mock

from collection import price_fetcher


def test_failed_download_uses_cached_prices_and_short_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(price_fetcher, "_CACHE_DIR", tmp_path)
    cached = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2020-01-02"], name="Date"),
    )
    cached.to_csv(tmp_path / "MSFT.csv")
    stale_time = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(tmp_path / "MSFT.csv", (stale_time, stale_time))
    download = Mock(return_value=pd.DataFrame())
    monkeypatch.setattr(price_fetcher.yf, "download", download)

    result = price_fetcher.fetch_price_history("MSFT", years=5)

    pd.testing.assert_frame_equal(result, cached)
    assert download.call_args.kwargs["timeout"] == 5
    assert download.call_args.kwargs["threads"] is False
