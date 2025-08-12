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
import re
import json
import base64
from pathlib import Path
from typing import Dict, Union, Optional

# API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Kept for non-vision fallbacks only (OpenAI/Gemini use image-only prompts)
SYSTEM_INSTR = (
    "You are an expert graph-reasoning assistant. "
    "Colour each vertex so no two adjacent vertices share a colour, "
    "using only Red, Blue, Green, or Yellow. "
    "Return exactly one line per vertex in the form:\n"
    "Vertex k: Colour \n"
    "Here is an example of a valid output format: {1: 'red', 2: 'blue', 3: 'green', 4: 'blue', 5: 'green', 6: 'yellow', 7: 'red', 8: 'blue'}\n"
    "You can use this EXACT format directly, no need to convert to a string, and no need to escape the curly braces."
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
            # OpenAI Python SDK v1 (Responses API)
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            if self.model:
                self.model = self._clean_text(self.model)

        elif self.provider == "gemini":
            if not GOOGLE_API_KEY:
                raise RuntimeError("Missing GOOGLE_API_KEY")
            # Google Gen AI SDK (google-genai)
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
            # DeepSeek is OpenAI-compatible
            from openai import OpenAI
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

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
        """Return a data URL suitable for the OpenAI image_url string."""
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    def _clean_text(self, text: str) -> str:
        """Make text ASCII-safe for headers/params (replace curly quotes/dashes)."""
        replacements = {
            "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
            "‘": "'", "’": "'", "‚": "'",
            "—": "-", "–": "-", "-": "-", "‒": "-"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("ascii", "ignore").decode("ascii")

    _PAIR_RE = re.compile(r"(?:vertex\s*)?(\d+)\s*[:=]\s*['\"]?\s*([A-Za-z]+)\s*['\"]?", re.IGNORECASE)

    def _parse_colouring(self, text: str) -> Dict[int, str]:
        """
        Parse a variety of model outputs into {int: colour}.
        Accepts lines like 'Vertex 1: red', dict-like '{1: red, 2: blue}', or JSON.
        """
        s = (text or "").strip()
        if not s:
            raise ValueError("empty")

        if s.upper().startswith("UNCOLORABLE"):
            raise ValueError("UNCOLORABLE")

        # Try JSON first
        if s[0] == "{":
            try:
                # Make a best-effort to coerce python-ish dicts into JSON:
                # - replace single quotes with double quotes when safe
                # - ensure keys are quoted
                j = s
                # Quote bare keys: {1: red} -> {"1": red}
                j = re.sub(r"{\s*", "{", j)
                j = re.sub(r"(\{|,)\s*(\d+)\s*:", r'\1 "\2":', j)
                # Quote bare string colours: red -> "red"
                j = re.sub(r':\s*([A-Za-z]+)\s*(?=[,}])', r': "\1"', j)
                # Convert single quotes to double quotes
                j = j.replace("'", '"')
                obj = json.loads(j)
                mapping: Dict[int, str] = {}
                for k, v in obj.items():
                    try:
                        kk = int(k)
                    except Exception:
                        continue
                    if isinstance(v, str):
                        mapping[kk] = v.strip().lower()
                if mapping:
                    return mapping
            except Exception:
                # Fall back to regex extraction
                pass

        # Regex extraction (handles both "Vertex 1: red" and "1: red")
        mapping: Dict[int, str] = {}
        for m in self._PAIR_RE.finditer(s):
            k = int(m.group(1))
            colour = m.group(2).strip().lower()
            mapping[k] = colour

        if not mapping:
            raise ValueError("No valid vertices parsed")
        return mapping

    # ---- OpenAI Responses text extraction (robust to reasoning parts) ----
    def _extract_openai_text(self, resp) -> str:
        """Return only the final textual answer from a Responses API response."""
        # 1) Easiest path:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        # 2) Defensive fallback: walk output items; ignore those without .content (e.g., reasoning)
        out_parts: list[str] = []
        for item in getattr(resp, "output", []) or []:
            parts = getattr(item, "content", None)
            if not parts:
                continue  # skip reasoning/tool items
            for part in parts:
                t = getattr(part, "type", "")
                if t in ("output_text", "text"):
                    out_parts.append(getattr(part, "text", "") or "")
        return "".join(out_parts).strip()

    # ─────────────── Main entrypoint ───────────────

    def prompt_for_colouring(
        self,
        image_path: Path,
        max_tokens: int = 256,
        temperature: float = 1.0
    ) -> Union[Dict[int, str], str]:
        """
        Prompt with *only the image* (instructions are drawn onto the PNG).
        OpenAI: Responses API with an input_image part (plus tiny text hint).
        Gemini: GenAI SDK with a bytes image part (no text).
        Others: simple text fallback (kept for completeness).
        Returns: dict {vertex:int -> colour:str} or "uncolorable".
        """
        # ── OpenAI (Responses API, image-only) ────────────────────────────────
        if self.provider == "openai":
            b64_url = self._encode_image_b64(image_path)
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": b64_url},
                            # Tiny nudge so models return just the mapping:
                            {"type": "input_text", "text": SYSTEM_INSTR}
                        ],
                    }
                ],
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            raw = self._extract_openai_text(resp)

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

        # Normalize & parse
        if "uncolorable" in (raw or "").lower():
            return raw

        try:
            mapping = self._parse_colouring(raw)
            # keep only allowed colours (normalize)
            norm = {int(k): str(v).lower() for k, v in mapping.items()}
            return norm + "\n\n Raw response: " + raw
        except Exception:
            return "invalid" + "\n\n Raw response: " + raw
