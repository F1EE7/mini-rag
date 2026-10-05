import importlib

app = importlib.import_module("app")


def test_end_to_end_offline(tmp_path, monkeypatch):
    docs = tmp_path / "资料"
    docs.mkdir()
    (docs / "math.md").write_text(
        "泰勒公式用多项式逼近复杂函数，是近似计算的重要工具。" * 3, encoding="utf-8"
    )
    (docs / "ball.md").write_text(
        "足球比赛每队十一人，在操场上进行对抗。" * 3, encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    chunks = app.build_index(str(docs))
    assert len(chunks) == 2
    assert {c["source"].split("#")[0] for c in chunks} == {"math.md", "ball.md"}

    hits = app.retrieve("足球 一共多少人", chunks, k=1)
    assert hits[0]["source"].startswith("ball.md")

    out = app.ask("泰勒公式有什么用？", chunks, k=2, offline=True)
    assert "离线" in out and "泰勒" in out
