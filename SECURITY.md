# Security

## Reporting a vulnerability

Please don't open a public issue. Use GitHub's private reporting instead:
**Security → Report a vulnerability** on https://github.com/pras-ops/Verdict. You should get a
reply within a few days.

## What the server protects

`verdict serve` can call paid model APIs with your keys and stores your prompts, so:

- It listens on `127.0.0.1` and accepts only `localhost` Host headers by default, which blocks
  DNS-rebinding attacks from web pages you visit.
- With `--token` (or `VERDICT_TOKEN`), every `/api/*`, `/v1/*` and `/metrics` request needs
  `Authorization: Bearer <token>`. `/api/health` and the static dashboard files stay open; they
  contain no data.
- Binding to another interface refuses to start without a token unless you pass `--no-auth`.
- `api_key_env` may only name variables ending in `_API_KEY` (or starting with `VERDICT_`), so a
  request can't make the server send other secrets to a URL of its choosing.
- Secret config values (`api_key`, `token`, `secret`, `password`) are masked in API responses.

## What it does not do

- There are no user accounts: anyone with the token has full access, including registering models
  that call any URL. Put Verdict behind your own authentication if several people share it.
- Traffic is plain HTTP. Use a reverse proxy with TLS for anything beyond your own machine.
- `verdict.db` holds prompts and answers in plain text.
