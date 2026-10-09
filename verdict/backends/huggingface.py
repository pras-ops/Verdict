from __future__ import annotations

from ..prompting import Prompt
from ..scoring import logsumexp
from .base import Backend, BackendError


class HuggingFaceBackend(Backend):
    """Open weights loaded in-process with transformers.

    mode="causal"     any chat/instruct LLM; exact probabilities over the full vocabulary
    mode="classifier" a model with a classification head (AutoModelForSequenceClassification);
                      its id2label names must match your option texts
    """

    kind = "huggingface"
    config_fields = {
        "model": ("str", "Qwen/Qwen2.5-0.5B-Instruct", "Hub id or local folder"),
        "mode": ("str", "causal", "causal (any LLM) or classifier (fine-tuned head)"),
        "device": ("str", "auto", "auto, cpu, cuda, mps"),
        "dtype": ("str", "auto", "auto, float16, bfloat16, float32"),
    }

    def __init__(self, **config):
        super().__init__(**config)
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise BackendError("install the extra: pip install 'verdict-llm[hf]'") from e
        name = self.config["model"]
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = self.config.get("dtype", "auto")
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        cls = AutoModelForSequenceClassification if self.config.get("mode") == "classifier" else AutoModelForCausalLM
        # transformers 4.56 renamed torch_dtype= to dtype=
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = cls.from_pretrained(name, **{dtype_kw: torch_dtype}).to(device).eval()
        self._device = device

    # ------------------------------------------------------------------
    def _render(self, prompt: Prompt) -> str:
        tok = self._tok
        if getattr(tok, "chat_template", None):
            try:
                text = tok.apply_chat_template(
                    prompt.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text = tok.apply_chat_template(prompt.messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(m["content"] for m in prompt.messages) + "\n\n"
        return text + prompt.prefill

    def label_logprobs(self, prompt: Prompt):
        with self.lock:
            self._load()
            if self.config.get("mode") == "classifier":
                return self._classify(prompt)
            return self._causal(prompt)

    def _causal(self, prompt: Prompt):
        import torch

        tok, model = self._tok, self._model
        ids = tok(self._render(prompt), return_tensors="pt", add_special_tokens=False).input_ids.to(self._device)
        with torch.no_grad():
            logp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)

        out: dict[str, float] = {}
        for label in prompt.labels:
            variants = []
            for text in {f" {label}", label}:
                t = tok.encode(text, add_special_tokens=False)
                if len(t) == 1:
                    variants.append(logp[t[0]].item())
                elif t:
                    variants.append(self._sequence_logprob(ids, t, logp))
            if variants:
                out[label] = logsumexp(variants)
        top = torch.topk(logp, 10)
        top_tokens = [(tok.decode([i]), v) for i, v in zip(top.indices.tolist(), top.values.tolist())]
        return out, top_tokens

    def _sequence_logprob(self, ids, cont: list[int], first_logp) -> float:
        import torch

        total = first_logp[cont[0]].item()
        full = torch.cat([ids, torch.tensor([cont], device=ids.device)], dim=1)
        with torch.no_grad():
            lp = torch.log_softmax(self._model(full).logits[0].float(), dim=-1)
        n = ids.shape[1]
        for j, t in enumerate(cont[1:], start=1):
            total += lp[n + j - 1, t].item()
        return total

    def _classify(self, prompt: Prompt):
        import torch

        # A fine-tuned classification head was trained on the raw text, not our chat prompt.
        text = prompt.input_text or prompt.messages[-1]["content"]
        enc = self._tok(text, return_tensors="pt", truncation=True).to(self._device)
        with torch.no_grad():
            logp = torch.log_softmax(self._model(**enc).logits[0].float(), dim=-1)
        id2label = {i: str(l) for i, l in self._model.config.id2label.items()}
        by_text = {l.lower(): logp[i].item() for i, l in id2label.items()}
        out = {}
        for label, option in zip(prompt.labels, prompt.options):
            if option.lower() in by_text:
                out[label] = by_text[option.lower()]
        if not out:
            raise BackendError(f"classifier labels {list(id2label.values())} do not match options {prompt.options}")
        return out, [(id2label[i], logp[i].item()) for i in id2label]

    def health(self) -> dict:
        try:
            with self.lock:
                self._load()
            return {"ok": True, "device": self._device}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def close(self) -> None:
        self._model = None
        self._tok = None
