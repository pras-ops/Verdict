from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="verdict", description="Turn any LLM into a decision engine.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the API and dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8420)
    s.add_argument("--db", default="verdict.db")

    d = sub.add_parser("decide", help="ask one question from the terminal")
    d.add_argument("question")
    d.add_argument("-o", "--option", action="append", default=[], help="repeat for each option (omit for yes/no)")
    d.add_argument("-c", "--context")
    d.add_argument("-b", "--backend", default="ollama")
    d.add_argument("-m", "--model", default="qwen3:4b")
    d.add_argument("--base-url", help="openai backend: server URL")
    d.add_argument("--scale", help="score on a scale, e.g. 1-5")
    d.add_argument("--debias", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn

        from .server import create_app

        print(f"Verdict dashboard: http://{args.host}:{args.port}")
        uvicorn.run(create_app(args.db), host=args.host, port=args.port, log_level="warning")
        return

    from .engine import Verdict

    cfg = {"model": args.model}
    if args.base_url:
        cfg["base_url"] = args.base_url
    v = Verdict.from_backend(args.backend, **cfg)
    if args.scale:
        lo, hi = (int(x) for x in args.scale.split("-"))
        r = v.score(args.question, context=args.context, scale=(lo, hi))
    elif args.option:
        r = v.choose(args.question, args.option, context=args.context, debias=args.debias)
    else:
        r = v.yes_no(args.question, context=args.context)
    print(json.dumps(r.model_dump(exclude={"top_tokens"}), indent=2))


if __name__ == "__main__":
    main()
