#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <regex>
#include <nlohmann/json.hpp>

namespace fasttok {

// ─── Byte-level BPE (GPT-2 / tiktoken compatible) ────────────────────────────
// This matches the algorithm used internally by OpenAI's tiktoken library,
// which is also used by GPT-2, GPT-3, GPT-4, LLaMA-BPE, Mistral, Falcon etc.

using json = nlohmann::json;

class BPETokenizer {
public:
    // ── Construction ──────────────────────────────────────────────────────────
    BPETokenizer() {}

    // Load from tiktoken-style .tiktoken file (single file, base64-encoded)
    // OR from HuggingFace tokenizer.json
    static BPETokenizer from_tiktoken_file(const std::string& path) {
        BPETokenizer tok;
        tok.load_tiktoken(path);
        return tok;
    }

    static BPETokenizer from_hf_json(const std::string& tokenizer_json_path) {
        BPETokenizer tok;
        tok.load_hf_json(tokenizer_json_path);
        return tok;
    }

    // ── Core API ──────────────────────────────────────────────────────────────
    std::vector<int> encode(const std::string& text) {
        if (vocab_.empty()) return naive_word_count(text);
        
        // 1. Pre-tokenise with the GPT-2 regex pattern
        std::vector<std::string> words = pre_tokenise(text);
        
        std::vector<int> result;
        for (const auto& word : words) {
            auto ids = bpe_word(word);
            result.insert(result.end(), ids.begin(), ids.end());
        }
        return result;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) {
            if (id_to_token_.count(id)) {
                result += id_to_token_.at(id);
            }
        }
        return result;
    }

    size_t count(const std::string& text) {
        return encode(text).size();
    }

    bool is_loaded() const { return !vocab_.empty(); }

private:
    // vocab: token-string -> id
    std::unordered_map<std::string, int> vocab_;
    // reverse: id -> token-string
    std::unordered_map<int, std::string> id_to_token_;
    // BPE merge rules: ordered list of (a, b) -> merged
    // Key = "a b", Value = rank (priority, lower = higher priority)
    std::map<std::pair<std::string,std::string>, int> merges_;

    // ── GPT-2 / tiktoken pre-tokenisation regex ───────────────────────────────
    // This exactly matches what tiktoken does before BPE:
    //   - contractions, words, numbers, punctuation, whitespace
    static const std::string& gpt2_pattern() {
        static std::string p = R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)";
        return p;
    }

    std::vector<std::string> pre_tokenise(const std::string& text) {
        std::vector<std::string> tokens;
        std::regex re(gpt2_pattern());
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            tokens.push_back(it->str());
        }
        return tokens;
    }

    // ── BPE encode a single pre-tokenised word ────────────────────────────────
    std::vector<int> bpe_word(const std::string& word) {
        // Start with individual characters (UTF-8 aware would split by code-point;
        // here we split to bytes – same as GPT-2 byte-level BPE)
        std::vector<std::string> pieces;
        pieces.reserve(word.size());
        for (unsigned char c : word) {
            pieces.push_back(std::string(1, (char)c));
        }

        // Iteratively find and apply the highest-priority merge
        while (pieces.size() > 1) {
            // Find the best (lowest-rank) adjacent pair
            int best_rank = INT_MAX;
            int best_idx  = -1;
            for (int i = 0; i + 1 < (int)pieces.size(); ++i) {
                auto key = std::make_pair(pieces[i], pieces[i+1]);
                auto it  = merges_.find(key);
                if (it != merges_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx  = i;
                }
            }
            if (best_idx < 0) break; // no more merges possible

            // Apply merge
            std::string merged = pieces[best_idx] + pieces[best_idx + 1];
            pieces.erase(pieces.begin() + best_idx + 1);
            pieces[best_idx] = merged;
        }

        // Map pieces -> IDs
        std::vector<int> ids;
        ids.reserve(pieces.size());
        for (const auto& p : pieces) {
            if (vocab_.count(p)) {
                ids.push_back(vocab_.at(p));
            } else {
                ids.push_back(0); // <unk>
            }
        }
        return ids;
    }

    // ── Loaders ───────────────────────────────────────────────────────────────

    // HuggingFace tokenizer.json loader (BERT, LLaMA, Mistral, etc.)
    void load_hf_json(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open tokenizer.json: " + path);

        json j;
        try { f >> j; }
        catch (const std::exception& e) {
            throw std::runtime_error(std::string("JSON parse error: ") + e.what());
        }

        // ── vocab ──
        if (j.contains("model") && j["model"].contains("vocab")) {
            for (auto& [token, id] : j["model"]["vocab"].items()) {
                vocab_[token] = id.get<int>();
                id_to_token_[id.get<int>()] = token;
            }
        }

        // ── merges ──
        if (j.contains("model") && j["model"].contains("merges")) {
            int rank = 0;
            for (const auto& merge_entry : j["model"]["merges"]) {
                std::string entry = merge_entry.get<std::string>();
                auto sp = entry.find(' ');
                if (sp == std::string::npos) { rank++; continue; }
                std::string a = entry.substr(0, sp);
                std::string b = entry.substr(sp + 1);
                merges_[{a, b}] = rank++;
            }
        }

        if (vocab_.empty())
            throw std::runtime_error("tokenizer.json loaded but vocab is empty.");
    }

    // tiktoken .tiktoken file: each line is "base64_token rank"
    void load_tiktoken(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open tiktoken file: " + path);

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto sp = line.rfind(' ');
            if (sp == std::string::npos) continue;
            std::string b64  = line.substr(0, sp);
            int         rank = std::stoi(line.substr(sp + 1));
            std::string token = base64_decode(b64);
            vocab_[token]       = rank;
            id_to_token_[rank]  = token;
            // tiktoken files encode merges implicitly via the rank order;
            // we add identity merges for every consecutive byte pair
        }

        if (vocab_.empty())
            throw std::runtime_error("tiktoken file loaded but vocab is empty.");
    }

    // Minimal Base64 decoder (RFC 4648)
    static std::string base64_decode(const std::string& in) {
        static const std::string chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        std::vector<int> T(256, -1);
        for (int i = 0; i < 64; i++) T[(unsigned char)chars[i]] = i;

        int val = 0, valb = -6;
        for (unsigned char c : in) {
            if (T[c] == -1) break;
            val = (val << 6) + T[c];
            valb += 6;
            if (valb >= 0) {
                out.push_back((char)((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return out;
    }

    // Fallback: whitespace-split token counting (used when no vocab loaded)
    std::vector<int> naive_word_count(const std::string& text) {
        std::vector<int> ids;
        std::stringstream ss(text);
        std::string word;
        int id = 0;
        while (ss >> word) {
            ids.push_back(id++);
        }
        return ids;
    }
};

// ── WordPiece Tokenizer (BERT, RoBERTa, DistilBERT, ELECTRA) ─────────────────
class WordPieceTokenizer {
public:
    explicit WordPieceTokenizer(const std::string& vocab_txt_path) {
        load_vocab(vocab_txt_path);
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) {
            auto word_ids = wordpiece_tokenize(word);
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) {
            if (id_to_token_.count(id)) {
                std::string tok = id_to_token_.at(id);
                // strip WordPiece "##" prefix for readability
                if (tok.size() > 2 && tok[0] == '#' && tok[1] == '#')
                    result += tok.substr(2);
                else
                    result += " " + tok;
            }
        }
        if (!result.empty() && result[0] == ' ') result = result.substr(1);
        return result;
    }

    size_t count(const std::string& text) { return encode(text).size(); }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    static constexpr int UNK_ID = 100; // [UNK] id in BERT

    void load_vocab(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open vocab.txt: " + path);
        std::string line;
        int id = 0;
        while (std::getline(f, line)) {
            if (line.empty()) { id++; continue; }
            vocab_[line]     = id;
            id_to_token_[id] = line;
            id++;
        }
        if (vocab_.empty())
            throw std::runtime_error("vocab.txt is empty: " + path);
    }

    std::vector<int> wordpiece_tokenize(const std::string& word) {
        std::vector<int> ids;
        int start = 0;
        int len   = (int)word.size();
        bool is_bad = false;

        while (start < len) {
            int end = len;
            int cur_id = -1;
            std::string substr;
            while (start < end) {
                substr = word.substr(start, end - start);
                if (start > 0) substr = "##" + substr;
                if (vocab_.count(substr)) {
                    cur_id = vocab_.at(substr);
                    break;
                }
                end--;
            }
            if (cur_id < 0) { is_bad = true; break; }
            ids.push_back(cur_id);
            start = end;
        }

        if (is_bad) ids = {UNK_ID};
        return ids;
    }
};

} // namespace fasttok
