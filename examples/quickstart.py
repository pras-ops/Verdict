"""The library in one page. Run with a local model:

    ollama pull qwen3:4b
    python examples/quickstart.py            # or: VERDICT_MODEL=qwen2.5:7b python examples/quickstart.py
"""
import os

from verdict import Verdict

v = Verdict.from_backend("ollama", model=os.environ.get("VERDICT_MODEL", "qwen3:4b"))

email = "URGENT: You won a $1,000,000 prize! Click this link within 24 hours to claim it."
r = v.choose("Which folder should this email go to?", ["Work", "Personal", "Spam"], context=email)
print(f"folder:   {r.choice}  ({r.confidence:.0%} sure, coverage {r.coverage:.0%})  {r.probs}")

ticket = "The production database is down and no customer can complete a payment."
print(f"urgent:   P(yes) = {v.yes_no('Is this urgent?', context=ticket).score:.2f}")

review = "Really good headphones, great sound. The case feels a little cheap though."
print(f"rating:   {v.score('How positive is this review?', context=review, scale=(1, 5)).value:.2f} / 5")

# Position bias: with no right answer, ask once per rotation of the options and average.
r = v.choose("Pick one at random.", ["Red", "Blue", "Green"], debias="rotate", min_confidence=0.8)
print(f"random:   {r.probs}  abstained={r.abstained}")

# Several questions about one input, in the Jev / System One format.
res = v.ask(
    "I was charged twice this month and nobody answers my emails. Refund the extra charge now.",
    {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, refunds", "technical": "Bugs and outages", "sales": "Pricing"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Annoyed", "Very angry"]},
        "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    },
)
for name, answer in res["answers"].items():
    print(f"{name + ':':12s}{answer}")
