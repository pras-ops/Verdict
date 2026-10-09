# Verdict

Turn **any LLM** into a decision engine — the Jev/Laya idea, but model-agnostic.
Instead of generating text, the model returns a **choice, a yes/no probability, or a score**,
read straight from its next-token probabilities. Comes with a Python library, a REST API,
a Grafana-style dashboard, and a Prometheus `/metrics` endpoint.

```
context + question + options ──► prompt with lettered options + "Answer:" prefill
                               ──► one forward pass, read P(" A"), P(" B"), ...
                               ──► normalise (+ calibration temperature) ──► {option: probability}
```

No text generation, no parsing: one token, real probabilities.

## Quick start

Needs Python 3.10+.

```bash
git clone https://github.com/pras-ops/Verdict.git
cd Verdict
pip install -e .            # add [hf] for Hugging Face: pip install -e ".[hf]"
python -m verdict serve     # dashboard on http://127.0.0.1:8420
```

For local models, install [Ollama](https://ollama.com) and pull one, e.g. `ollama pull qwen3:4b`.

Then open the **Models** tab, add a model, and try it in **Playground**.

From the terminal:

```bash
python -m verdict decide "Which folder should this go to?" -o Work -o Personal -o Spam -c "You won a prize, click here" -m qwen3:4b
```

## Library

```python
from verdict import Verdict

v = Verdict.from_backend("ollama", model="qwen3:4b")

r = v.choose("Which folder?", ["Work", "Personal", "Spam"], context=email)
r.choice, r.probs, r.confidence            # 'Spam', {'Work': 0.001, ...}, 0.999

v.yes_no("Is this urgent?", context=ticket).score        # P(yes)
v.score("How angry is the customer?", context=msg, scale=(1, 5)).value   # expected rating
v.choose(..., debias=True)                 # ask twice with options reversed, average (fixes position bias)
```

## Backends

| backend | config | probabilities |
|---|---|---|
| `ollama` | `model`, `host` | top-20 logprobs (Ollama ≥ 0.12.11) |
| `openai` | `model`, `base_url`, `api_key_env` | top-20 logprobs — OpenAI, vLLM, LM Studio, OpenRouter, Groq, llama.cpp server … |
| `huggingface` | `model`, `mode=causal\|classifier`, `device` | **exact**, full vocabulary; `classifier` mode uses a fine-tuned classification head (the Laya approach) |
| `mock` | `sharpness`, `latency_ms` | fake, for tests and demos |

If a model can't return logprobs (e.g. OpenAI reasoning models), Verdict falls back to
**constrained output** (JSON schema with an enum) and marks the result `method="constrained"`.

Add your own connector:

```python
from verdict import Backend, register_backend

@register_backend
class MyBackend(Backend):
    kind = "mine"
    def top_logprobs(self, prompt, k=20):
        ...  # return [(token, logprob), ...] for the first generated token
```

## Reading the result

- `probs` — probability per option, sums to 1
- `confidence` — probability of the chosen option
- `coverage` — how much of the model's probability landed on valid labels at all.
  Low coverage (< 50%) means the model wanted to say something else — treat the answer with care.
- `method` — `logprobs` (real probabilities) or `constrained` (fallback, 0/1 only)

## Calibration

Raw LLM probabilities are usually over-confident. In **Evaluate**, paste labelled examples
(one JSON per line), run them, and Verdict fits a **temperature** that minimises log-loss.
Click *Apply temp.* and every later decision from that model is calibrated.
You can also label live decisions on the Overview page to track real-world accuracy.

## API

| method | path | |
|---|---|---|
| POST | `/api/decide` | `{model, kind, question, options?, context?, scale?, debias?}` |
| POST | `/api/compare` | same, with `models: [...]` — runs them in parallel |
| POST | `/api/evaluate` | `{models, examples}` → accuracy, log-loss, ECE, fitted temperature |
| GET/POST/DELETE | `/api/models` | register models |
| POST | `/api/models/{name}/test` | health check + probe decision |
| PUT | `/api/models/{name}/temperature` | set calibration |
| GET | `/api/decisions`, `/api/stats` | log + aggregates |
| POST | `/api/decisions/{id}/feedback` | `{label}` ground truth |
| GET | `/metrics` | Prometheus |

## Real Grafana

Point Prometheus at Verdict and build Grafana panels on:
`verdict_decisions_total`, `verdict_errors_total`, `verdict_constrained_total`,
`verdict_correct_total / verdict_labelled_total` (live accuracy), `verdict_latency_ms_bucket` (histogram).

```yaml
scrape_configs:
  - job_name: verdict
    static_configs: [{ targets: ["127.0.0.1:8420"] }]
```

## Limits (v0.1)

- Choice supports up to 26 options (A–Z); scores are single-digit scales (0–9).
- `top_logprobs` APIs cap at 20 tokens; options outside the top 20 get an upper-bound estimate.
- Chat models answer much better than base models (see `coverage`).

## Tests

```bash
pytest -q
```
