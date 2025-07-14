"""
llm.py
Unified multimodal colouring interface for:
 • openai    (Vision‐chat via inline base64 blocks)
 • claude     (Anthropic; text‐only)
 • gemini     (Google; HTTP‐fetched image‐URL)
 • deepseek   (text‐only)
 • llama      (local HF; text‐only embed)
"""

from __future__ import annotations
import os
import base64
from pathlib import Path
from typing import Dict, Union

# ─── READ YOUR API KEYS FROM THE ENVIRONMENT ────────────────────────────────────
OPENAI_API_KEY    = "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"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY    = "AIzaSyBqIvR2AqhYVSSI0tO6XCC6hMRq4UIB2iU"
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
# ────────────────────────────────────────────────────────────────────────────────
SYSTEM_INSTR = (
    "You are an expert graph‐reasoning assistant. "
    "Colour each vertex so no two adjacent vertices share a colour, "
    "using only Red, Blue, Green, or Yellow. "
    "Return exactly one line per vertex in the form:\n"
    "Vertex <k>: <Colour>\n"
    "If it cannot be 4-coloured, reply exactly with: UNCOLORABLE"
)

class LLMColourer:
    def __init__(self, provider: str, model: str | None = None, device: int = -1) -> None:
        self.provider = provider.lower()
        self.model = model
        self.device = device

        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("Missing OPENAI_API_KEY")
            import openai
            openai.api_key = OPENAI_API_KEY
            self.client = openai

        elif self.provider == "gemini":
            if not GOOGLE_API_KEY:
                raise RuntimeError("Missing GOOGLE_API_KEY")
            # Official Google Gen AI SDK
            from google import genai
            self.client = genai.Client(api_key=GOOGLE_API_KEY)

        elif self.provider == "claude":
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("Missing ANTHROPIC_API_KEY")
            from anthropic import AnthropicClient
            self.client = AnthropicClient(api_key=ANTHROPIC_API_KEY)

        elif self.provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise RuntimeError("Missing DEEPSEEK_API_KEY")
            import deepseek
            deepseek.api_key = DEEPSEEK_API_KEY
            self.client = deepseek

        elif self.provider == "llama":
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            if not model:
                raise ValueError("Need --model for llama provider")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto" if device >= 0 else None,
                torch_dtype="auto"
            )
            self.client = pipeline(
                "text-generation",
                model=self.model_obj,
                tokenizer=self.tokenizer,
                device=device
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _encode_image(self, image_path: Path) -> str:
        """Base64‐encode a PNG file for inline embedding (OpenAI only)."""
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    def _parse_response(self, text: str) -> Dict[int, str]:
        text = text.strip()
        if text.upper() == "UNCOLORABLE":
            raise ValueError("UNCOLORABLE")

        mapping: Dict[int, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            idx, col = line.split(":", 1)
            try:
                v = int(idx.strip().split()[-1])
                mapping[v] = col.strip().lower()
            except ValueError:
                continue

        if not mapping:
            raise ValueError("No valid vertices parsed")
        return mapping

    def prompt_for_colouring(
        self,
        image_ref: Path,
        max_tokens: int = 256,
        temperature: float = 1.0
    ) -> Union[Dict[int, str], str]:
        """
        Sends only the graph PNG to the chosen backend:
         - OpenAI: inline base64
         - Gemini:   embeds raw bytes via Part.from_bytes()
         - Others:   text‐only
        """
        raw: str

        # ─── OpenAI Vision ─────────────────────────────────────────────
        if self.provider == "openai":
            b64 = self._encode_image(image_ref)
            prompt = [
                {"type": "text",      "text": SYSTEM_INSTR},
                {"type": "image_url", "image_url": {"url": b64}}
            ]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            raw = resp.choices[0].message.content

        # ─── Google Gemini (Gen AI SDK) ─────────────────────────────────
        elif self.provider == "gemini":
            from google.genai import types

            # 1. Read the PNG bytes
            img_bytes = image_ref.read_bytes()

            # 2. Build a true multimodal prompt
            contents = [
                types.Part.from_text(text=SYSTEM_INSTR),
                types.Part.from_text(
                    text="Please colour the graph in the provided image. "
                         "Return one line per vertex, 'Vertex k: <Colour>'."
                ),
                types.Part.from_bytes(
                    content=img_bytes,
                    mime_type="image/png"
                )
            ]

            # 3. Call the Gemini model
            resp = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            raw = resp.text

        # ─── Claude (Anthropic) ────────────────────────────────────────
        elif self.provider == "claude":
            prompt = SYSTEM_INSTR + "\n\n(Graph image provided separately)"
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens_to_sample=max_tokens,
                temperature=temperature
            )
            raw = resp["completion"]

        # ─── DeepSeek (text‐only) ───────────────────────────────────────
        elif self.provider == "deepseek":
            prompt = SYSTEM_INSTR + "\n\n(Graph image provided separately)"
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",  "content": SYSTEM_INSTR},
                    {"role": "user",    "content": prompt}
                ],
                temperature=temperature
            )
            raw = resp.choices[0].message.content

        # ─── LLaMA (local HF; text‐only) ───────────────────────────────
        else:  # llama
            prompt = SYSTEM_INSTR + "\n\n(Graph image provided separately)"
            out = self.client(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=temperature
            )
            raw = out[0]["generated_text"]
            if raw.startswith(prompt):
                raw = raw[len(prompt):].strip()

        # ─── Short‐circuit “uncolorable” ───────────────────────────────
        if "uncolorable" in raw.lower():
            return "uncolorable"

        # ─── Parse & return colouring dict ─────────────────────────────
        try:
            return self._parse_response(raw)
        except ValueError:
            return "uncolorable"