# FastTok

**High-performance C++ tokenizer and prompt compressor for LLMs.**
Reduce OpenAI, Claude, and Gemini API costs by 10–25% with one line of code.

![Tests](https://github.com/Anique/fasttok/actions/workflows/tests.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/fasttok.svg)](https://badge.fury.io/py/fasttok)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why FastTok?

Every token you send to an LLM API costs money. FastTok's C++ compression engine
automatically rewrites your prompts before they're sent — removing verbose phrases
and collapsing multi-word expressions — reducing token usage by **10–25%** without
changing the meaning.

**Verified results (tiktoken exact counts):**

| Text Type | Before | After | Saved |
|---|---|---|---|
| Verbose AI instruction | 62 tokens | 51 tokens | **17.7%** |
| Short tech prompt | 27 tokens | 23 tokens | **14.8%** |
| Business email | 58 tokens | 54 tokens | **6.9%** |

---

## Installation

```bash
pip install fasttok
```

> Requires a C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS).  
> Pre-built wheels available for Windows, Linux, macOS (Python 3.8–3.12).

---

## Quick Start

```python
from fasttok import Tokenizer

# Works with any LLM — auto-detects algorithm
tok = Tokenizer.from_pretrained("gpt-4o")        # OpenAI
tok = Tokenizer.from_pretrained("claude-3-5-sonnet")  # Anthropic
tok = Tokenizer.from_pretrained("gemini-1.5-pro")     # Google
tok = Tokenizer.from_pretrained("llama-3")             # Meta / Groq
tok = Tokenizer.from_pretrained("bert-base-uncased")   # BERT (WordPiece)

text = "For your information, the software development is in order to meet the requirements."

# Compress before sending to API (saves tokens = saves money)
compressed = tok.compress(text)
# "FYI, the software dev is to meet the reqs."

# Count tokens (with optional budget warning)
n = tok.count(text, budget=20)

# Decompress AI response back to human-readable English
readable = tok.decompress("FYI the app dev is ASAP done")
# "for your information the application development is as soon as possible done"

# Trim to fit exactly within a context window
trimmed = tok.smart_trim(text, max_tokens=10)
```

### Real OpenAI Integration

```python
from openai import OpenAI
from fasttok import Tokenizer

tok  = Tokenizer.from_pretrained("gpt-4o-mini")
client = OpenAI(api_key="sk-...")

prompt = "For your information, the application development is approximately done."
compressed = tok.compress(prompt)   # "FYI, the app dev is approximately done."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": compressed}]
)
print(f"Tokens saved: {tok.count(prompt) - tok.count(compressed)}")
```

---

## Supported Models

| Provider | Models | Algorithm |
|---|---|---|
| OpenAI | GPT-4, GPT-4o, GPT-3.5, o1, o3 | BPE (tiktoken) |
| Anthropic | Claude 3 / 4 / Sonnet / Haiku | BPE (HF) |
| Google | Gemini 1.5, Gemma | BPE (HF) |
| Meta | LLaMA 2 / 3 | BPE (HF) |
| Mistral AI | Mistral, Mixtral | BPE (HF) |
| Groq | Groq (LLaMA-based) | BPE (HF) |
| TII | Falcon 40B | BPE (HF) |
| Alibaba | Qwen 2.5 | BPE (HF) |
| DeepSeek | DeepSeek Coder | BPE (HF) |
| Google | BERT, DistilBERT, ELECTRA | WordPiece |
| HuggingFace | RoBERTa, ALBERT | WordPiece |

**Exact BPE Parity**: supply a `tokenizer.json` or `.tiktoken` file:
```python
tok = Tokenizer.from_pretrained("llama-3", vocab="path/to/tokenizer.json")
```

---

## Compression Strategies

1. **Multi-word → single token**: `"in order to"` → `"to"`, `"due to the fact that"` → `"because"`
2. **Acronym expansion**: `"as soon as possible"` → `"ASAP"`, `"for your information"` → `"FYI"`
3. **Technical phrase reduction**: `"application development"` → `"app dev"`
4. **Whitespace/newline normalization**: collapse redundant spaces and newlines
5. **Bidirectional**: `decompress()` re-expands for human-readable output

---

## Performance

- **C++17 core** with pybind11 — runs GIL-free
- **AVX2 SIMD + OpenMP** compiled in by default
- **Zero network calls** at runtime — fully offline after install
- **Python 3.8–3.12** supported

---

## License

[MIT](LICENSE) © 2026 Anique
