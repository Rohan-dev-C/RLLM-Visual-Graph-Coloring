"""
llm.py
Unified colouring interface that sends *images only* to vision-capable models.
 • openai   – Responses API with image parts (inline base64)
 • gemini   – Google Gen AI SDK: image-only (bytes part)
 • claude   – text-only (fallback)
 • deepseek – text-only (fallback)
 • llama    – local HF text-only (fallback)
"""
from __future__ import annotations
import os
import base64
from pathlib import Path
from typing import Dict, Union

# API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")

# Kept for non-vision fallbacks only (OpenAI/Gemini use image-only prompts)
SYSTEM_INSTR = (
    "You are an expert graph-reasoning assistant. "
    "Colour each vertex so no two adjacent vertices share a colour, "
    "using only Red, Blue, Green, or Yellow. "
    "Return exactly one line per vertex in the form:\n"
    "Vertex k: Colour \n"
    "Here is an example of a valid output format: {1: 'red', 2: 'blue', 3: 'green', 4: 'blue', 5: 'green', 6: 'yellow', 7: 'red', 8: 'blue'}\n"
    "You can use the format directly, no need to convert to a string, and no need to escape the curly braces."
    "No need to write 'red' or 'blue' or 'green' or 'yellow', just write the colour name."
    "If you try to write the colour name in quotes, you will be penalised."
    "Use the format to directly return the colouring, do not write any other text."
    "Do not explain, just return the colouring."
    "Your response will be parsed using the JSON.loads function, so ensure the format is correct."
    "Use a raw string literal for the format, do not use any escapes."
    "If you fail to return the correct format, you will be penalised."
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
            # Use the modern Responses API client per OpenAI docs.
            # https://platform.openai.com/docs/api-reference/responses
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            # Ensure model name is ASCII-safe
            if self.model:
                self.model = self._clean_text(self.model)

        elif self.provider == "gemini":
            if not GOOGLE_API_KEY:
                raise RuntimeError("Missing GOOGLE_API_KEY")
            # Google Gen AI SDK
            from google import genai
            self.genai = genai.Client(api_key=GOOGLE_API_KEY)

        elif self.provider == "claude":
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("Missing ANTHROPIC_API_KEY")
            from anthropic import AnthropicClient
            self.client = AnthropicClient(api_key=ANTHROPIC_API_KEY)

        elif self.provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise RuntimeError("Missing DEEPSEEK_API_KEY")
            # Use OpenAI-compatible client for DeepSeek API
            from openai import OpenAI
            self.client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1"
            )

        elif self.provider == "llama":
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            if not self.model:
                raise ValueError("Need --model for llama provider")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                self.model, device_map="auto" if device >= 0 else None, torch_dtype="auto"
            )
            self.client = pipeline("text-generation", model=self.model_obj, tokenizer=self.tokenizer, device=device)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ─────────────── Helpers ───────────────

    def _encode_image_b64(self, image_path: Path) -> str:
        """Return a data URL suitable for the OpenAI image_url object."""
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    def _clean_text(self, text: str) -> str:
        """Clean text to ensure it's ASCII-safe for API calls."""
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        # Replace em dashes and en dashes
        text = text.replace('—', '-').replace('–', '-')
        # Ensure it's ASCII-safe
        return text.encode('ascii', 'ignore').decode('ascii')

    def _parse_response(self, text: str) -> Dict[int, str]:
        text = text.strip()
        if text.upper() == "UNCOLORABLE":
            raise ValueError("UNCOLORABLE")
        mapping: Dict[int, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            try:
                k = int(left.strip().split()[-1])
                mapping[k] = right.strip().lower()
            except ValueError:
                continue
        if not mapping:
            raise ValueError("No valid vertices parsed")
        return mapping

    # ─────────────── Main entrypoint ───────────────

    def prompt_for_colouring(
        self,
        image_path: Path,
        max_tokens: int = 256,
        temperature: float = 1.0
    ) -> Union[Dict[int, str], str]:
        """
        Prompt with *only the image* (instructions are drawn onto the PNG).
        OpenAI: Responses API with an input_image part (no text).
        Gemini: GenAI SDK with a bytes image part (no text).
        Others: simple text fallback (kept for completeness).
        """
        # ── OpenAI (Responses API, image-only) ────────────────────────────────
        # ── OpenAI (Responses API, image-only) ────────────────────────────────
        # ── OpenAI (Responses API, image-only) ────────────────────────────────
        if self.provider == "openai":
            # Build a data URL string and pass it directly as a string (not an object)
            b64_url = self._encode_image_b64(image_path)
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": b64_url  # <-- string, not {"url": ...}
                            },
                            {
                                "type": "input_text",
                                "text": SYSTEM_INSTR
                            }
                        ],
                    }
                ],
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            # Try the convenient accessor first; fall back to walking the parts
            raw = getattr(resp, "output_text", "") or ""
            if not raw and getattr(resp, "output", None):
                for part in resp.output[0].content:
                    if getattr(part, "type", "") == "output_text":
                        raw += part.text


        # ── Google Gemini (Gen AI SDK; image-only bytes) ──────────────────────
        elif self.provider == "gemini":
            from google.genai import types
            img_bytes = image_path.read_bytes()
            contents = [types.Part.from_bytes(content=img_bytes, mime_type="image/png")]
            resp = self.genai.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            raw = resp.text or ""

        # ── Claude / DeepSeek / LLaMA: text-only fallback ─────────────────────
        elif self.provider == "claude":
            prompt = SYSTEM_INSTR + "\n\n(Graph image instructions are embedded on the image.)"
            resp = self.client.completions.create(
                model=self.model, prompt=prompt, max_tokens_to_sample=max_tokens, temperature=temperature
            )
            raw = resp["completion"]

        elif self.provider == "deepseek":
            prompt = SYSTEM_INSTR + "\n\n(Graph image instructions are embedded on the image.)"
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            raw = resp.choices[0].message.content

        else:  # llama
            prompt = SYSTEM_INSTR + "\n\n(Graph image instructions are embedded on the image.)"
            out = self.client(prompt, max_new_tokens=max_tokens, do_sample=False, temperature=temperature)
            raw = out[0]["generated_text"]
            if raw.startswith(prompt):
                raw = raw[len(prompt):].strip()


        return raw
