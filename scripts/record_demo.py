"""Record docs/demo.gif: a real run of the dashboard against local Ollama models.

    pip install playwright            # uses your installed Chrome; no browser download
    ollama pull qwen2.5:7b && ollama pull qwen2.5:14b
    verdict serve --port 8422 --db /tmp/demo.db     # in another terminal
    python scripts/record_demo.py                   # needs ffmpeg on PATH

Everything on screen is live model output; nothing is mocked. Time spent waiting for the
models is fast-forwarded in the GIF (8x) so it stays short.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent

PHISHING = ("From: IT-Helpdesk <it-support@micros0ft-security.co>\n"
            "Your work password expires today. Log in here to keep access: http://bit.ly/x9a2")
REVIEW = "Really good headphones, great sound. The case feels a little cheap though."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8422")
    ap.add_argument("--models", nargs="+", default=["qwen2.5:7b", "qwen2.5:14b"])
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--fast-forward", type=float, default=8.0, help="speed-up while waiting for models")
    args = ap.parse_args()

    for tag in args.models:  # register the models through the API
        name = tag.replace(":", "-")
        httpx.post(f"{args.url}/api/models", json={"name": name, "backend": "ollama", "config": {"model": tag}}).raise_for_status()
        httpx.post(f"{args.url}/api/models/{name}/test", timeout=300)  # load into memory so the demo isn't waiting

    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 820}, color_scheme="dark",
                                  record_video_dir=str(tmp), record_video_size={"width": 1280, "height": 820})
        page = ctx.new_page()
        t0 = time.monotonic()
        waits: list[tuple[float, float]] = []  # (start, end) seconds into the video

        def waited(start: float) -> None:
            waits.append((start + 0.4, time.monotonic() - t0))

        page.add_init_script("try { localStorage.clear() } catch {}")
        page.goto(args.url)
        page.wait_for_timeout(1200)

        # 1. Playground: two models judge the same phishing email, side by side
        page.click("nav button[data-tab=playground]")
        page.wait_for_timeout(600)
        page.fill("#pg-context", "")
        page.type("#pg-context", PHISHING, delay=12)
        page.fill("#pg-question", "")
        page.type("#pg-question", "Which folder should this email go to?", delay=20)
        for box in page.query_selector_all("#pg-models input"):
            box.check()
        page.wait_for_timeout(400)
        w = time.monotonic() - t0
        page.click("#pg-run")
        page.wait_for_selector("#pg-results .bars", timeout=120_000)
        waited(w)
        page.wait_for_timeout(300)
        page.evaluate("document.querySelector('#pg-results').scrollIntoView({behavior: 'smooth', block: 'end'})")
        page.wait_for_timeout(3200)

        # 2. A 1-5 score, read with full coverage
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(600)
        page.select_option("#pg-kind", "score")
        page.fill("#pg-context", "")
        page.type("#pg-context", REVIEW, delay=12)
        page.fill("#pg-question", "")
        page.type("#pg-question", "How positive is this review?", delay=20)
        w = time.monotonic() - t0
        page.click("#pg-run")
        page.wait_for_function("document.querySelector('#pg-results').innerText.includes('Score')", timeout=120_000)
        waited(w)
        page.wait_for_timeout(300)
        page.evaluate("document.querySelector('#pg-results').scrollIntoView({behavior: 'smooth', block: 'end'})")
        page.wait_for_timeout(3000)

        # 3. Evaluate both models on 36 labelled examples, calibrate
        page.click("nav button[data-tab=evaluate]")
        page.wait_for_timeout(600)
        page.click("#ev-load")
        page.wait_for_timeout(900)
        # one model: comparing two big local models at once makes Ollama swap them in and out of memory
        for i, box in enumerate(page.query_selector_all("#ev-models input")):
            box.set_checked(i == 0)
        page.check("#ev-log")
        w = time.monotonic() - t0
        page.click("#ev-run")
        page.wait_for_selector("#ev-results-panel:not([hidden])", timeout=300_000)
        waited(w)
        page.wait_for_timeout(500)
        page.evaluate("document.querySelector('#ev-results-panel').scrollIntoView({behavior: 'smooth', block: 'center'})")
        page.wait_for_timeout(3500)

        # 4. Overview: live accuracy, latency and every logged decision
        page.click("nav button[data-tab=overview]")
        page.wait_for_timeout(1500)
        page.mouse.move(900, 300)
        page.wait_for_timeout(1500)
        page.evaluate("window.scrollTo({top: 600, behavior: 'smooth'})")
        page.wait_for_timeout(2500)

        video = page.video.path()
        ctx.close()
        browser.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # normal speed while things happen on screen, fast-forward while waiting for the models
    cuts, t = [], 0.0
    for a, b in waits:
        if b - a > 1.0:
            cuts += [(t, a, 1.0), (a, b, args.fast_forward)]
            t = b
    cuts.append((t, None, 1.0))
    parts = [f"[0:v]trim=start={a:.2f}" + (f":end={b:.2f}" if b else "") + f",setpts=(PTS-STARTPTS)/{s}[v{i}]"
             for i, (a, b, s) in enumerate(cuts)]
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(cuts))) + f"concat=n={len(cuts)}:v=1:a=0[c];" + (
        f"[c]fps=10,scale={args.width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", graph,
                    "-loop", "0", str(out)], check=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
