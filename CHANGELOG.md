# Changelog

## 0.2.0 (2026-10-09)

### Fixed
- **Score questions read the wrong probabilities on Ollama.** Qwen, Llama 3 and GPT-4o-style
  tokenizers write a bare space after "Answer:" before the digit, so v0.1 read the 1–5 scores
  from near-zero noise (coverage ~0%). Verdict now looks one token further when the model's next
  token is only whitespace and merges both steps. On qwen2.5:7b, score coverage went from 0% to
  100% and benchmark accuracy from 80.6% to 83.3%.
- **OpenAI reasoning models failed instead of falling back.** The OpenAI-compatible backend now
  adapts to the server's errors and remembers them per model: it drops `temperature`, switches
  `max_tokens` to `max_completion_tokens`, sends `reasoning_effort="none"` so gpt-5.1+ / GPT-6
  return logprobs, lowers `top_logprobs`, and drops `response_format` where it isn't supported.
  Models that always reason (gpt-5, o-series) fall back to constrained output with a token budget
  that leaves room for reasoning, and a clear error if they still run out.
- **API keys could leak** to anyone who could reach the server: `api_key_env` could name any
  environment variable and `base_url` any URL, `GET /api/models` returned stored keys, and any
  `Host` header was accepted (DNS rebinding). See *Security* below.
- **Calibration could recommend a harmful temperature** from a handful of examples (the built-in
  5-example sample suggested 0.2, making an over-confident model more confident). It now needs at
  least 30 examples with some mistakes before the dashboard offers *Apply*, leaves
  constrained (0/1) answers out of the fit, keeps the temperature within 0.25–20, and accepts
  answers in any case ("yes" for "Yes").
- The server's worker pool was closed at shutdown and never recreated, so an app started a second
  time (test clients, reloading servers) failed every parallel request with
  "cannot schedule new futures after shutdown".
- Hugging Face classifier mode now feeds the raw text to the classification head instead of the
  formatted chat prompt; transformers ≥ 4.56 gets `dtype=` instead of the deprecated `torch_dtype=`.
- The CLI prints one clear error line instead of a Python traceback.

### Added
- **Jev / System One compatible endpoint** `POST /v1/systemone`: one state, several named
  `noul` / `choice` / `score` questions, same response shape as TypeSafe's Jev API. Also in Python
  as `Verdict.ask(state, questions)`.
- **`systemone` backend** to use Jev, Laya, Ollama decision models or another Verdict server as a
  model, with Verdict's calibration, debiasing, evaluation and monitoring on top.
- **MCP server** (`verdict mcp`) with `choose`, `yes_no`, `score` and `ask` tools; extra `[mcp]`.
- **`verdict batch`** classifies a CSV or JSONL file in parallel; `Verdict.decide_many()`.
- `debias="rotate"` asks once per rotation of the options, which cancels position bias fully.
- `min_confidence` flags uncertain results with `abstained=True`.
- Dashboard: provider presets, a token prompt, a "Load the 36-example benchmark" button, a
  warning instead of *Apply* when calibration evidence is thin, and the new debias mode.
- `verdict decide -q` prints only the answer; `--set key=value`, `--api-key-env`, `--temperature`,
  `--version`; `VERDICT_BACKEND` / `VERDICT_MODEL` / `VERDICT_BASE_URL` environment defaults.
- `GET /api/health`.
- Live tests against real models (`pytest -m live`, `pytest -m hf`), simulated-response tests for
  every backend, CI on Python 3.10–3.13, `examples/benchmark.py`, `scripts/record_demo.py`.
- MIT license.

### Security
- The server only accepts `localhost` Host headers by default (`--allowed-host` to change).
- `--token` / `VERDICT_TOKEN` protects `/api/*`, `/v1/*` and `/metrics`; a non-loopback
  `--host` refuses to start without one unless `--no-auth` is given.
- `api_key_env` must name a variable ending in `_API_KEY` (or starting with `VERDICT_`).
- Secret config values are masked as `***` in API responses.

### Changed
- Published as **`verdict-llm`** (the name `verdict` on PyPI belongs to another project). The import
  name is still `verdict`.
- The OpenAI preset defaults to `gpt-4.1-mini`.

## 0.1.0

First release: choice / yes-no / score decisions from token probabilities; Ollama, OpenAI-compatible,
Hugging Face and mock backends; constrained-output fallback; temperature calibration; position
debiasing; REST API, dashboard, SQLite log and Prometheus metrics.
