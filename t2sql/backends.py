"""SQL generators. Each backend exposes generate(messages) -> (raw_text, seconds)
and describe() -> dict of what actually ran. Heavy imports stay inside the classes
so the scorer and its tests never need torch."""

from __future__ import annotations

import time

MAX_NEW_TOKENS = 256
OLLAMA_URL = "http://localhost:11434"
OLLAMA_NUM_CTX = 4096


class GoldBackend:
    """Returns the gold SQL. Used to smoke-test the harness: it must score 100%."""

    name = "gold-oracle"

    def generate_for(self, example: dict) -> tuple[str, float]:
        return example["gold"], 0.0

    def describe(self) -> dict:
        return {"backend": "gold"}


class ConstantBackend:
    """Always returns the same query. Used to smoke-test that wrong SQL scores as wrong."""

    name = "constant"

    def __init__(self, sql: str = "SELECT 1"):
        self.sql = sql

    def generate_for(self, example: dict) -> tuple[str, float]:
        return self.sql, 0.0

    def describe(self) -> dict:
        return {"backend": "constant", "sql": self.sql}


class HFBackend:
    """transformers on the local GPU in fp16, greedy decoding, one question at a time.
    With adapter_path, the LoRA weights are merged into the base model before timing."""

    def __init__(self, model_id: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path).merge_and_unload()
        self.model.to("cuda").eval()
        self.generate([{"role": "user", "content": "SELECT"}], max_new_tokens=4)  # warm-up

    def generate(self, messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS):
        torch = self.torch
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        text = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, seconds

    def describe(self) -> dict:
        torch = self.torch
        return {
            "backend": "transformers",
            "model": self.model_id,
            "adapter": self.adapter_path,
            "dtype": "float16",
            "decoding": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        }


class OllamaBackend:
    """A model served by a local Ollama daemon, temperature 0."""

    def __init__(self, model: str, url: str = OLLAMA_URL):
        import requests

        self.requests = requests
        self.model = model
        self.url = url
        self.generate([{"role": "user", "content": "SELECT"}], max_new_tokens=4)  # load + warm-up

    def generate(self, messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS):
        start = time.perf_counter()
        r = self.requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "seed": 0,
                    "num_predict": max_new_tokens,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            },
            timeout=600,
        )
        seconds = time.perf_counter() - start
        r.raise_for_status()
        return r.json()["message"]["content"], seconds

    def unload(self):
        self.requests.post(
            f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=60
        )

    def describe(self) -> dict:
        info = {"backend": "ollama", "model": self.model, "decoding": "temperature 0",
                "max_new_tokens": MAX_NEW_TOKENS, "num_ctx": OLLAMA_NUM_CTX}
        try:
            show = self.requests.post(
                f"{self.url}/api/show", json={"model": self.model}, timeout=30
            ).json()
            info["details"] = show.get("details")
            ps = self.requests.get(f"{self.url}/api/ps", timeout=30).json()
            for m in ps.get("models", []):
                if m.get("name") == self.model:
                    info["size_bytes"] = m.get("size")
                    info["size_vram_bytes"] = m.get("size_vram")
        except Exception as e:
            info["describe_error"] = str(e)
        return info
