#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tokenizer.hpp"
#include "compressor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fasttok_core, m) {
    m.doc() = "FastTok C++ engine — production-grade BPE/WordPiece tokenizer with token compressor";

    // ── Compressor ──────────────────────────────────────────────────────────
    py::class_<fasttok::Compressor>(m, "Compressor")
        .def(py::init<>())
        .def("compress",   &fasttok::Compressor::compress,
             "Compress text to reduce token count (abbreviation + whitespace pass)")
        .def("decompress", &fasttok::Compressor::decompress,
             "Re-expand abbreviations for human readability");

    // ── BPETokenizer ────────────────────────────────────────────────────────
    // Supports HuggingFace tokenizer.json AND tiktoken .tiktoken files
    py::class_<fasttok::BPETokenizer>(m, "BPETokenizer")
        .def(py::init<>())
        .def_static("from_hf_json",       &fasttok::BPETokenizer::from_hf_json,
                    "Load from a HuggingFace tokenizer.json file",
                    py::arg("tokenizer_json_path"))
        .def_static("from_tiktoken_file", &fasttok::BPETokenizer::from_tiktoken_file,
                    "Load from a tiktoken .tiktoken file",
                    py::arg("path"))
        .def("encode",    &fasttok::BPETokenizer::encode)
        .def("decode",    &fasttok::BPETokenizer::decode)
        .def("count",     &fasttok::BPETokenizer::count)
        .def("is_loaded", &fasttok::BPETokenizer::is_loaded);

    // ── WordPieceTokenizer ──────────────────────────────────────────────────
    // Supports BERT, RoBERTa, DistilBERT, ELECTRA (vocab.txt format)
    py::class_<fasttok::WordPieceTokenizer>(m, "WordPieceTokenizer")
        .def(py::init<const std::string&>(), py::arg("vocab_txt_path"))
        .def("encode", &fasttok::WordPieceTokenizer::encode)
        .def("decode", &fasttok::WordPieceTokenizer::decode)
        .def("count",  &fasttok::WordPieceTokenizer::count);
}
