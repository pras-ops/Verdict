from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from pydantic import ValidationError

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-b", "--backend", default=os.environ.get("VERDICT_BACKEND", "ollama"),
                   help="ollama, openai, huggingface or mock (env VERDICT_BACKEND)")
    p.add_argument("-m", "--model", default=os.environ.get("VERDICT_MODEL", "qwen3:4b"),
                   help="model name on that backend (env VERDICT_MODEL)")
    p.add_argument("--base-url", default=os.environ.get("VERDICT_BASE_URL"),
                   help="server URL: Ollama host, or an OpenAI-compatible /v1 URL (env VERDICT_BASE_URL)")
    p.add_argument("--api-key-env", help="openai backend: env var holding the API key (default OPENAI_API_KEY)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="any other backend setting, e.g. --set reasoning_effort=minimal (repeatable)")
    p.add_argument("--temperature", type=float, default=1.0, help="calibration temperature (default 1.0)")


def _value(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return s


def _engine(args):
    from .engine import Verdict

    cfg = {"model": args.model}
    if args.base_url:
        cfg["host" if args.backend == "ollama" else "base_url"] = args.base_url
    if args.api_key_env:
        cfg["api_key_env"] = args.api_key_env
    for item in args.set:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        cfg[key.strip()] = _value(val)
    return Verdict.from_backend(args.backend, temperature=args.temperature, **cfg)


def _scale(s: str) -> tuple[int, int]:
    lo, sep, hi = s.partition("-")
    if not sep:
        raise ValueError(f"--scale expects LOW-HIGH such as 1-5, got '{s}'")
    return int(lo), int(hi)


def _fail(msg: str) -> None:
    print(f"verdict: error: {msg}", file=sys.stderr)
    sys.exit(1)


def _explain(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
    return str(e)


# -- commands -----------------------------------------------------------------
def cmd_serve(args) -> None:
    import uvicorn

    from .server import create_app

    token = args.token or os.environ.get("VERDICT_TOKEN") or None
    loopback = args.host in LOOPBACK
    if not loopback and not token and not args.no_auth:
        _fail(
            f"refusing to listen on {args.host} without a token, because anyone on the network could use "
            "your models and API keys. Pass --token (or set VERDICT_TOKEN), or --no-auth if you trust this network."
        )
    hosts = args.allowed_host or (["localhost", "127.0.0.1", "::1"] if loopback else ["*"])
    app = create_app(args.db, token=token, allowed_hosts=hosts, default_model=args.default_model)
    shown = "localhost" if loopback else args.host
    print(f"Verdict dashboard: http://{shown}:{args.port}" + ("  (token required)" if token else ""))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_decide(args) -> None:
    v = _engine(args)
    try:
        if args.scale:
            r = v.score(args.question, context=args.context, scale=_scale(args.scale))
        elif args.option:
            r = v.choose(args.question, args.option, context=args.context, debias=args.debias)
        else:
            r = v.yes_no(args.question, context=args.context)
    finally:
        v.close()
    if args.quiet:
        print(r.choice)
    else:
        print(json.dumps(r.model_dump(exclude={"top_tokens", "id"}), indent=2))


def _read_rows(path: str, field: str) -> list[dict]:
    if path == "-":
        return _parse_jsonl(sys.stdin, "stdin", field)
    with open(path, newline="", encoding="utf-8") as fh:
        if not path.endswith(".csv"):
            return _parse_jsonl(fh, path, field)
        rows = list(csv.DictReader(fh))
    for n, row in enumerate(rows, 2):
        if None in row:  # DictReader puts surplus fields under the key None
            raise ValueError(f"{path} line {n} has more columns than the header; put quotes around text that contains commas")
    return rows


def _parse_jsonl(fh, name: str, field: str) -> list[dict]:
    rows = []
    for n, line in enumerate(fh, 1):
        if line.strip():
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{name} line {n} is not valid JSON: {e}") from None
            rows.append(obj if isinstance(obj, dict) else {field: obj})
    return rows


def cmd_batch(args) -> None:
    """Classify every row of a JSONL or CSV file, in parallel."""
    rows = _read_rows(args.input, args.field)
    if not rows:
        _fail(f"{args.input} has no rows")
    if args.question is None and args.input.endswith(".csv"):
        _fail("CSV input needs --question (and -o/--option for a choice); the text comes from --field")

    def to_decision(row: dict) -> dict:
        if args.question is None:  # each JSONL line is a full decision
            return {k: v for k, v in row.items() if k not in ("id", "answer")}
        d = {"question": args.question, "context": str(row.get(args.field, "")), "debias": args.debias}
        if args.scale:
            d.update(kind="score", scale=_scale(args.scale))
        elif args.option:
            d.update(kind="choice", options=args.option)
        else:
            d["kind"] = "binary"
        return d

    v = _engine(args)
    try:
        decisions = []
        for row in rows:
            try:
                decisions.append(to_decision(row))
            except Exception as e:  # noqa: BLE001
                decisions.append(e)
        results = v.decide_many(
            [d for d in decisions if not isinstance(d, Exception)], workers=args.workers, return_exceptions=True
        )
    finally:
        v.close()
    it = iter(results)
    out_rows = []
    for row, d in zip(rows, decisions):
        r = d if isinstance(d, Exception) else next(it)
        out = dict(row)
        if isinstance(r, Exception):
            out["error"] = _explain(r)
        else:
            out.update(choice=r.choice, confidence=round(r.confidence, 4),
                       coverage=None if r.coverage is None else round(r.coverage, 4), method=r.method)
            if r.kind == "binary":
                out["p_yes"] = round(r.score, 4)
            if r.kind == "score":
                out["value"] = round(r.value, 4)
            if args.probs:
                out["probs"] = {k: round(p, 4) for k, p in r.probs.items()}
        out_rows.append(out)
    _write_rows(out_rows, args.output)
    failed = sum(1 for o in out_rows if "error" in o)
    print(f"verdict: {len(out_rows) - failed} decided, {failed} failed", file=sys.stderr)


def _write_rows(rows: list[dict], path: str) -> None:
    if path == "-":
        _dump(rows, sys.stdout, as_csv=False)
    else:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            _dump(rows, fh, as_csv=path.endswith(".csv"))


def _dump(rows: list[dict], fh, as_csv: bool) -> None:
    if as_csv:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    else:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_mcp(args) -> None:
    from .mcp_server import run

    run(_engine(args))


# -- entry point --------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="verdict", description="Turn any LLM into a decision engine.")
    from . import __version__

    ap.add_argument("--version", action="version", version=f"verdict {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the API and dashboard")
    s.add_argument("--host", default="127.0.0.1", help="interface to listen on (default 127.0.0.1, this computer only)")
    s.add_argument("--port", type=int, default=8420)
    s.add_argument("--db", default="verdict.db", help="SQLite file for models and the decision log")
    s.add_argument("--token", help="require this token on every API call (env VERDICT_TOKEN)")
    s.add_argument("--allowed-host", action="append", metavar="HOST",
                   help="accepted Host header (repeatable); default: localhost names, or any when --host is public")
    s.add_argument("--default-model", help="registered model answering /v1/systemone requests for unknown models")
    s.add_argument("--no-auth", action="store_true", help="allow a public --host without a token (not recommended)")
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("decide", help="ask one question from the terminal")
    d.add_argument("question")
    d.add_argument("-o", "--option", action="append", default=[], help="repeat for each option (omit for yes/no)")
    d.add_argument("-c", "--context", help="the text the decision is about")
    d.add_argument("--scale", help="rate on a scale instead, e.g. 1-5")
    d.add_argument("--debias", action="store_true", help="also ask with options reversed (2 calls)")
    d.add_argument("-q", "--quiet", action="store_true", help="print only the answer")
    _add_model_args(d)
    d.set_defaults(func=cmd_decide)

    b = sub.add_parser("batch", help="classify every row of a JSONL or CSV file")
    b.add_argument("input", help="file.jsonl or file.csv ('-' for JSONL on stdin)")
    b.add_argument("--output", default="-", help="where to write results: .jsonl or .csv (default stdout)")
    b.add_argument("--question", help="ask this of every row (otherwise each JSONL line is a full decision)")
    b.add_argument("-o", "--option", action="append", default=[], help="with --question: repeat for each option")
    b.add_argument("--scale", help="with --question: rate on a scale, e.g. 1-5")
    b.add_argument("--field", default="text", help="column/key holding the text (default 'text')")
    b.add_argument("--debias", action="store_true")
    b.add_argument("--probs", action="store_true", help="include the full probability table per row")
    b.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    _add_model_args(b)
    b.set_defaults(func=cmd_batch)

    m = sub.add_parser("mcp", help="run an MCP server so AI agents can ask for decisions (stdio)")
    _add_model_args(m)
    m.set_defaults(func=cmd_mcp)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .backends import BackendError

    try:
        args.func(args)
    except (BackendError, ValidationError, ValueError, OSError) as e:
        _fail(_explain(e))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
