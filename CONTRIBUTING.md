# Contributing

Thanks for helping. Bug reports, new backends and benchmark results on other models are all welcome.

## Set up

```bash
git clone https://github.com/pras-ops/Verdict.git
cd Verdict
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
ruff check .
pytest                                          # fast, no model needed
VERDICT_LIVE_MODEL=qwen2.5:7b pytest -m live    # if you touched a backend, the engine or the server
```

- Add a test for every fix. Backend behaviour is tested against simulated server responses
  (`httpx.MockTransport`, see `tests/test_backends.py`), so tests stay fast and need no network.
- If you change how probabilities are read, run `python examples/benchmark.py <model>` before and
  after, and put both results in the pull request.
- Keep the dashboard dependency-free: plain HTML, CSS and JavaScript, no build step.
- Update `CHANGELOG.md` under an "Unreleased" heading.

## Adding a backend

Subclass `verdict.Backend`, set `kind` and `config_fields`, and implement `top_logprobs`
(first-token log-probabilities) and/or `constrained_choice` (fallback). Override `label_logprobs`
if the service already returns a probability per answer (see `backends/systemone.py`). Register it
in `backends/__init__.py`, add a preset to the dashboard (`PRESETS` in `dashboard/app.js`), and
list it in the README's provider table with what was actually tested.

## Releasing

1. Bump `version` in `pyproject.toml` and `verdict/__init__.py`; move the changelog entries under the new version.
2. `python -m build` and check the wheel installs in a fresh environment.
3. Tag `vX.Y.Z` and upload with `twine upload dist/*`.
