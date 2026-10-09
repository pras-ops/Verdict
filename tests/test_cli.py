import csv
import json

import pytest

from verdict.cli import main

MOCK = ["-b", "mock", "-m", "mock", "--set", "latency_ms=0"]


def run(capsys, *argv):
    with pytest.raises(SystemExit) as e:
        main(list(argv))
        raise SystemExit(0)
    out = capsys.readouterr()
    return e.value.code, out.out, out.err


def test_decide_prints_json_or_just_the_answer(capsys):
    code, out, _ = run(capsys, "decide", "invoice question", "-o", "invoice", "-o", "holiday", *MOCK)
    assert code == 0 and json.loads(out)["choice"] == "invoice"
    code, out, _ = run(capsys, "decide", "invoice question", "-o", "invoice", "-o", "holiday", "-q", *MOCK)
    assert out.strip() == "invoice"


def test_errors_are_one_clear_line(capsys):
    code, out, err = run(capsys, "decide", "rate it", "--scale", "1-10", *MOCK)
    assert code == 1 and err.startswith("verdict: error: scale must be single digits") and "Traceback" not in err
    code, _, err = run(capsys, "decide", "q", "--base-url", "http://127.0.0.1:9", "--set", "timeout=2")
    assert code == 1 and "cannot reach Ollama" in err
    code, _, err = run(capsys, "decide", "q", "--set", "oops", *MOCK)
    assert code == 1 and "KEY=VALUE" in err


def test_serve_refuses_a_public_host_without_a_token(capsys, monkeypatch):
    monkeypatch.delenv("VERDICT_TOKEN", raising=False)
    code, _, err = run(capsys, "serve", "--host", "0.0.0.0")
    assert code == 1 and "without a token" in err


def test_batch_csv_with_a_question(tmp_path, capsys):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("text\nurgent invoice attached\nholiday photos from the beach\n")
    code, _, err = run(capsys, "batch", str(src), "--question", "Which folder?", "-o", "urgent invoice", "-o", "holiday photos",
                       "--output", str(dst), *MOCK)
    rows = list(csv.DictReader(dst.open()))
    assert code == 0 and "2 decided, 0 failed" in err
    assert [r["choice"] for r in rows] == ["urgent invoice", "holiday photos"]
    assert {"text", "choice", "confidence", "coverage", "method"} <= set(rows[0])


def test_batch_jsonl_of_full_decisions_reports_bad_rows(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(d) for d in [
        {"id": 1, "question": "red apple", "options": ["red apple", "blue sky"], "answer": "red apple"},
        {"id": 2, "question": "broken", "options": ["only one"]},
        {"id": 3, "question": "Is it on?", "kind": "binary"},
    ]))
    code, out, err = run(capsys, "batch", str(src), "--probs", *MOCK)
    rows = [json.loads(line) for line in out.splitlines()]
    assert code == 0 and "2 decided, 1 failed" in err
    assert rows[0]["id"] == 1 and rows[0]["choice"] == "red apple" and "probs" in rows[0]
    assert "at least 2 options" in rows[1]["error"]
    assert 0 <= rows[2]["p_yes"] <= 1


def test_batch_csv_needs_a_question(tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text("text\nhello\n")
    code, _, err = run(capsys, "batch", str(src), *MOCK)
    assert code == 1 and "needs --question" in err


def test_batch_csv_with_unquoted_commas_is_explained(tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text("text\nhello, world\n")
    code, _, err = run(capsys, "batch", str(src), "--question", "q", *MOCK)
    assert code == 1 and "put quotes around text" in err
