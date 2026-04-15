#pragma once
#include <string>
#include <regex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace fasttok {

class Compressor {
public:
    Compressor() {
        // Rules: only include phrases where the SHORT form has FEWER tokens than the FULL form
        // Avoid abbreviations that add punctuation (e.g., "approx." splits as 2 tokens)
        abbrev_list = {
            // Multi-word -> short single word (always saves tokens)
            {" for your information ", " FYI "},
            {" as soon as possible ",  " ASAP "},
            {" by the way ",           " BTW "},
            {" with respect to ",      " about "},
            {" in addition to ",       " plus "},
            {" in addition, ",         " also, "},
            {" please note that ",     " note: "},
            {" thank you for ",        " thanks for "},
            {" thanks for your ",      " thanks for your "},
            // Application / tech multi-word reductions
            {" application development ", " app dev "},
            {" the application ",      " the app "},
            {" software development ", " software dev "},
            {" the requirements ",     " the reqs "},
            {" the reference ",        " the ref "},
            {" the information ",      " the info "},
            {" the library ",          " the lib "},
            {" the libraries ",        " the libs "},
            // Multi-word phrases -> single word
            {" as well as ",           " and "},
            {" in order to ",          " to "},
            {" make sure that ",       " ensure "},
            {" in the event that ",   " if "},
            {" at this point in time ", " now "},
            {" due to the fact that ", " because "},
            {" a large number of ",   " many "},
        };
        // Build reverse map for decompression
        for (const auto& [full, abbr] : abbrev_list) {
            expand_list.push_back({abbr, full});
        }
    }

    std::string compress(const std::string& input) {
        if (input.empty()) return input;
        // Pad with spaces so phrases at start/end of string are matched
        std::string result = " " + input + " ";

        // 1. Whitespace: collapse tabs/spaces
        result = std::regex_replace(result, std::regex("[ \t]+"), " ");
        // 2. Newline: collapse multiple newlines
        result = std::regex_replace(result, std::regex("[\r\n]+"), "\n");
        // 3. Abbreviation pass (ordered: longest first)
        for (const auto& [full, abbr] : abbrev_list) {
            size_t pos = 0;
            while ((pos = result.find(full, pos)) != std::string::npos) {
                result.replace(pos, full.length(), abbr);
                pos += abbr.length();
            }
        }
        // 4. Trim leading/trailing whitespace
        auto start = result.find_first_not_of(" \t\n\r");
        auto end   = result.find_last_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        return result.substr(start, end - start + 1);
    }

    std::string decompress(const std::string& input) {
        if (input.empty()) return input;
        std::string result = input;
        for (const auto& [abbr, full] : expand_list) {
            size_t pos = 0;
            while ((pos = result.find(abbr, pos)) != std::string::npos) {
                result.replace(pos, abbr.length(), full);
                pos += full.length();
            }
        }
        return result;
    }

private:
    // List of (full_form, abbreviation) pairs — ORDER MATTERS (longest first)
    std::vector<std::pair<std::string, std::string>> abbrev_list;
    // Reverse for decompression: (abbreviation, full_form)
    std::vector<std::pair<std::string, std::string>> expand_list;
};

} // namespace fasttok
