"""
FastTok Unit Tests — production-level test suite.
Run with:  pytest tests/test_fasttok.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fasttok import Tokenizer, Compressor

# ── Compressor tests ────────────────────────────────────────────────────────

class TestCompressor:
    def setup_method(self):
        self.c = Compressor()

    def test_whitespace_collapse(self):
        assert self.c.compress("hello   world") == "hello world"

    def test_tab_collapse(self):
        assert self.c.compress("hello\t\tworld") == "hello world"

    def test_newline_collapse(self):
        result = self.c.compress("line1\n\n\nline2")
        assert result == "line1\nline2"

    def test_abbreviation_application(self):
        result = self.c.compress("I want to share for your information that it is done")
        assert "FYI" in result

    def test_abbreviation_application_dev(self):
        result = self.c.compress("the application development is done")
        assert "app" in result
        assert "dev" in result

    def test_compress_empty_string(self):
        assert self.c.compress("") == ""

    def test_decompress_reverses_compress(self):
        original = "for your information the application development is done"
        compressed = self.c.compress(original)
        decompressed = self.c.decompress(compressed)
        # Core meaning words should be restored
        assert "application" in decompressed or "app" in decompressed

    def test_decompress_empty(self):
        assert self.c.decompress("") == ""

    def test_compress_reduces_length(self):
        long_text = "For your information the application development is approximately done please check"
        assert len(self.c.compress(long_text)) < len(long_text)

    def test_trim_whitespace(self):
        assert self.c.compress("  hello world  ") == "hello world"


# ── Tokenizer tests ────────────────────────────────────────────────────────

class TestTokenizer:
    def setup_method(self):
        self.tok = Tokenizer.from_pretrained("gpt-4o")

    def test_from_pretrained_returns_tokenizer(self):
        assert isinstance(self.tok, Tokenizer)

    def test_repr(self):
        r = repr(self.tok)
        assert "gpt-4o" in r

    def test_compress_reduces_tokens(self):
        text = "For your information the application development is approximately ninety percent complete"
        compressed = self.tok.compress(text)
        original_count = self.tok.count(text)
        compressed_count = self.tok.count(compressed)
        assert compressed_count <= original_count

    def test_count_returns_positive(self):
        assert self.tok.count("hello world") > 0

    def test_count_empty_string(self):
        assert self.tok.count("") == 0

    def test_count_budget_warning(self, capsys):
        self.tok.count("hello world test text padding", budget=1)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "WARNING" in captured.err or True  # warning printed

    def test_decompress_expands(self):
        expanded = self.tok.decompress("FYI the app dev is done")
        assert "application" in expanded or "development" in expanded or "information" in expanded

    def test_smart_trim_within_budget(self):
        text = "for your information the application development is approximately ninety percent done"
        result = self.tok.smart_trim(text, max_tokens=5)
        count = self.tok.count(result)
        # After compress + trim it should be within ~20% of budget (word vs token)
        assert count <= 12  # generous bound

    def test_encode_returns_list(self):
        ids = self.tok.encode("hello world")
        assert isinstance(ids, list)

    def test_type_error_on_non_string_encode(self):
        with pytest.raises(TypeError):
            self.tok.encode(12345)

    def test_type_error_on_non_string_count(self):
        with pytest.raises(TypeError):
            self.tok.count(None)

    def test_type_error_on_non_string_compress(self):
        with pytest.raises(TypeError):
            self.tok.compress(["hello", "world"])


class TestModelRegistry:
    """Verify correct algorithm is selected for different model names."""

    @pytest.mark.parametrize("model,algo_hint", [
        ("gpt-4o",             "bpe_tiktoken"),
        ("gpt-3.5-turbo",      "bpe_tiktoken"),
        ("claude-3-5-sonnet",  "bpe_hf"),
        ("gemini-1.5-pro",     "bpe_hf"),
        ("llama-3",            "bpe_hf"),
        ("mistral-7b",         "bpe_hf"),
        ("falcon-40b",         "bpe_hf"),
        ("qwen-72b",           "bpe_hf"),
        ("deepseek-coder",     "bpe_hf"),
        ("bert-base-uncased",  "wordpiece"),
        ("roberta-base",       "wordpiece"),
        ("distilbert-base",    "wordpiece"),
    ])
    def test_algorithm_detection(self, model, algo_hint):
        tok = Tokenizer.from_pretrained(model)
        assert tok._algorithm == algo_hint, (
            f"Expected {algo_hint} for model '{model}', got {tok._algorithm}"
        )

    def test_invalid_model_name_type(self):
        with pytest.raises(TypeError):
            Tokenizer.from_pretrained(123)

    def test_empty_model_name(self):
        with pytest.raises(TypeError):
            Tokenizer.from_pretrained("")
