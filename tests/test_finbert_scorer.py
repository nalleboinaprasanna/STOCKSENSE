from sentiment import finbert_scorer


def test_score_finbert_batches_headlines(monkeypatch):
    calls = []

    def pipeline(batch, *, batch_size):
        calls.append((batch, batch_size))
        return [{"label": "positive", "score": 0.8} for _ in batch]

    monkeypatch.setattr(finbert_scorer, "_load_pipeline", lambda: pipeline)

    headlines = [f"Headline {i}" for i in range(33)]
    scores = finbert_scorer.score_finbert(headlines)

    assert len(scores) == len(headlines)
    assert scores == [0.8] * len(headlines)
    assert [len(batch) for batch, _ in calls] == [16, 16, 1]
    assert all(batch_size == 16 for _, batch_size in calls)
