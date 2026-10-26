"""Pruebas unitarias de las mejoras, sin Ollama, ChromaDB ni red."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiagnostico(unittest.TestCase):
    def test_summarize_records(self):
        m = load_module("diag", "scripts/diagnostico_rag.py")
        result = m.summarize_records([
            {"text": "uno", "page": None}, {"text": "cuatro", "page": 2}
        ])
        self.assertEqual(result["fragmentos"], 2)
        self.assertEqual(result["fragmentos_con_pagina"], 1)
        self.assertEqual(result["caracteres_por_fragmento"]["maximo"], 6)

    def test_empty(self):
        m = load_module("diag2", "scripts/diagnostico_rag.py")
        self.assertEqual(m.summarize_records([])["fragmentos"], 0)


class TestEvaluacion(unittest.TestCase):
    def setUp(self):
        self.m = load_module("evalRAG", "scripts/evaluar_rag.py")
        self.case = {"expected_source": "dataflow.pdf", "expected_terms": ["Watermarks"]}
        self.first = SimpleNamespace(source="data/raw/dataflow.pdf", page=7, text="Texto general")
        self.second = SimpleNamespace(source="data/raw/dataflow.pdf", page=8, text="Watermarks en streaming")

    def test_matching_source_and_text(self):
        self.assertFalse(self.m.is_relevant(self.first, self.case))
        self.assertTrue(self.m.is_relevant(self.second, self.case))
        self.assertEqual(self.m.first_relevant_rank([self.first, self.second], self.case), 2)

    def test_page_criterion(self):
        with_page = {**self.case, "expected_page": 7}
        self.assertFalse(self.m.is_relevant(self.second, with_page))

    def test_summary(self):
        rows = [
            {"method": "hybrid", "error": "", "hit_at_k": 1, "reciprocal_rank": 1.0, "latency_ms": 10.0},
            {"method": "hybrid", "error": "", "hit_at_k": 0, "reciprocal_rank": 0, "latency_ms": 20.0},
        ]
        result = self.m.summary_by_method(rows)["hybrid"]
        self.assertEqual(result["hit_at_k"], .5)
        self.assertEqual(result["mrr"], .5)
        self.assertEqual(result["latencia_media_ms"], 15)

    def test_two_cases_required(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cases.json"
            p.write_text('[{"question":"uno", "expected_source":"x.pdf"}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                self.m.load_cases(p)


if __name__ == "__main__":
    unittest.main()
