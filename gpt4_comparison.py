"""
GPT-4o-Mini Token Comparison Test
Uses tiktoken (OpenAI's official library) for EXACT token counts.
Install:  pip install tiktoken
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fasttok import Tokenizer

MODEL = "gpt-4o-mini"

# ── Try tiktoken for exact counts ─────────────────────────────────────────────
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
    COUNT_MODE = "EXACT  (tiktoken)"
except ImportError:
    import re
    _PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+"
    def count_tokens(text: str) -> int:
        return len(re.findall(_PATTERN, text))
    COUNT_MODE = "ESTIMATE (install tiktoken for exact counts)"

def sep(title=""):
    print("=" * 58)
    if title:
        print(f"  {title}")
        print("=" * 58)

def run():
    sep(f"FastTok -- GPT-4o-Mini Token Compression Test")
    print(f"  Count mode : {COUNT_MODE}")
    tok = Tokenizer.from_pretrained(MODEL)
    print()

    prompts = [
        (
            "Short tech prompt",
            "For your information, the application development is as soon as possible done. "
            "Please check the requirements and reference documents in addition to the library."
        ),
        (
            "Verbose AI instruction",
            "In order to make sure that the output is correct, due to the fact that the model "
            "does not always follow instructions, at this point in time we need to ensure "
            "compliance. A large number of tokens are wasted. In the event that this happens, "
            "please note that we should try again as soon as possible."
        ),
        (
            "Business email",
            "Thank you for your message. As soon as possible, we will review the requirements "
            "with respect to the application development timeline. In addition to this effort, "
            "the software development team will also check the reference documentation. "
            "Please note that a large number of updates are expected by the end of the week."
        ),
    ]

    total_orig = total_comp = 0

    for label, prompt in prompts:
        compressed   = tok.compress(prompt)
        orig_n = count_tokens(prompt)
        comp_n = count_tokens(compressed)
        saved  = orig_n - comp_n
        pct    = saved / orig_n * 100 if orig_n else 0
        total_orig  += orig_n
        total_comp  += comp_n

        sep(label)
        print(f"  ORIGINAL  ({orig_n} tokens):")
        print(f"    {prompt[:100]}{'...' if len(prompt)>100 else ''}")
        print(f"\n  FASTTOK   ({comp_n} tokens):")
        print(f"    {compressed[:100]}{'...' if len(compressed)>100 else ''}")
        print(f"\n  [Saved {saved} tokens / {pct:.1f}% reduction]\n")

    sep("TOTAL RESULTS")
    total_saved = total_orig - total_comp
    total_pct   = total_saved / total_orig * 100 if total_orig else 0
    print(f"  Original total  : {total_orig} tokens")
    print(f"  Compressed total: {total_comp} tokens")
    print(f"  Tokens saved    : {total_saved}")
    print(f"  Cost reduction  : {total_pct:.1f}%")
    print("=" * 58)

if __name__ == "__main__":
    run()
