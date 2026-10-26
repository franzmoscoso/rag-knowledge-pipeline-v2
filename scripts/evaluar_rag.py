"""Evaluación reproducible de recuperación RAG sin alterar el pipeline existente.

Compara búsqueda semántica, BM25, híbrida y opcionalmente RRF.
Mide Hit@k y MRR contra criterios documentales definidos por el grupo.
Opcionalmente genera UNA respuesta por pregunta con el método híbrido.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize(value: str) -> str:
    cleaned = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in cleaned if not unicodedata.combining(char))


def is_relevant(chunk, case: dict) -> bool:
    """Relevancia según fuente + página opcional + uno de los términos esperados.

    No sustituye la revisión humana de la pertinencia semántica.
    """
    expected_source = (case.get("expected_source") or "").strip()
    if not expected_source:
        return False
    actual = Path(str(chunk.source).replace("\\", "/")).name
    reference = Path(expected_source.replace("\\", "/")).name
    if normalize(actual) != normalize(reference):
        return False
    if case.get("expected_page") is not None and int(chunk.page or -1) != int(case["expected_page"]):
        return False
    terms = case.get("expected_terms") or []
    if terms and not any(normalize(t) in normalize(chunk.text) for t in terms):
        return False
    return True


def first_relevant_rank(chunks: list, case: dict) -> int | None:
    return next((i for i, c in enumerate(chunks, 1) if is_relevant(c, case)), None)


def summary_by_method(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for method in sorted({r["method"] for r in rows}):
        valid = [r for r in rows if r["method"] == method and not r.get("error")]
        all_method = [r for r in rows if r["method"] == method]
        n = len(valid)
        summary[method] = {
            "casos_evaluados": n,
            "errores": len(all_method) - n,
            "hit_at_k": round(sum(r["hit_at_k"] for r in valid) / n, 4) if n else None,
            "mrr": round(sum(r["reciprocal_rank"] for r in valid) / n, 4) if n else None,
            "latencia_media_ms": round(sum(r["latency_ms"] for r in valid) / n, 2) if n else None,
        }
    return summary


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Se requieren al menos 2 casos de evaluación en un arreglo JSON.")
    for i, case in enumerate(data, 1):
        if not isinstance(case, dict) or not case.get("question") or not case.get("expected_source"):
            raise ValueError(f"Caso {i}: se requieren 'question' y 'expected_source'.")
        if not isinstance(case.get("expected_terms", []), list):
            raise ValueError(f"Caso {i}: 'expected_terms' debe ser una lista de frases.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa la recuperación RAG y guarda evidencias JSON/CSV.")
    parser.add_argument("--cases", default="data/casos_evaluacion.json")
    parser.add_argument("--space", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output-dir", default="evidencias")
    parser.add_argument("--rrf", action="store_true", help="Incluir RRF como cuarto método")
    parser.add_argument("--generate", action="store_true", help="Generar 1 respuesta por pregunta usando híbrida; requiere Ollama")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k debe ser mayor que 0")
    cases = load_cases(Path(args.cases))

    from rag.retriever import Retriever
    from rag.vector_store import VectorStore
    from config.settings import settings

    store = VectorStore(space=args.space)
    if store.count() == 0:
        parser.error("La colección está vacía. Ejecute primero 'python cli.py ingest --source ... --reset'.")
    indexed_sources = {normalize(Path(str(x["source"]).replace("\\", "/")).name) for x in store.list_sources()}
    missing_sources = sorted({case["expected_source"] for case in cases
                              if normalize(Path(case["expected_source"].replace("\\", "/")).name)
                              not in indexed_sources})
    if missing_sources:
        parser.error("Fuentes esperadas no indexadas: " + ", ".join(missing_sources)
                     + ". Verifique nombres en el JSON y vuelva a ejecutar 'cli.py ingest'.")
    retriever = Retriever(store, use_graph=False, use_rerank=False)
    methods = {
        "semantic": retriever.semantic_search,
        "bm25": retriever.keyword_search,
        "hybrid": retriever.hybrid_search,
    }
    if args.rrf:
        methods["rrf"] = retriever.hybrid_rrf_search

    rows = []
    detailed = []
    for i, case in enumerate(cases, 1):
        question = case["question"]
        for method_name, run_search in methods.items():
            start = time.perf_counter()
            chunks, error = [], None
            try:
                chunks = run_search(question, args.top_k)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            rank = first_relevant_rank(chunks, case) if not error else None
            row = {
                "case_id": case.get("id", f"P{i}"),
                "question": question,
                "method": method_name,
                "top_k": args.top_k,
                "results": len(chunks),
                "first_relevant_rank": rank if rank is not None else "",
                "hit_at_k": int(rank is not None),
                "reciprocal_rank": round(1 / rank, 4) if rank else 0,
                "latency_ms": elapsed,
                "error": error or "",
            }
            rows.append(row)
            detail = {
                **row,
                "expected_source": case["expected_source"],
                "expected_page": case.get("expected_page"),
                "expected_terms": case.get("expected_terms", []),
                "retrieved": [
                    {
                        "rank": pos,
                        "source": c.source,
                        "page": c.page,
                        "chunk_id": c.chunk_id,
                        "method": c.method,
                        "score": round(float(c.score), 6),
                        "relevant_by_criteria": is_relevant(c, case),
                        "excerpt": c.text[:500],
                    }
                    for pos, c in enumerate(chunks, 1)
                ],
            }
            if args.generate and method_name == "hybrid" and not error:
                from rag.answering import answer_question
                try:
                    result = answer_question(
                        question, retriever, top_k=args.top_k, agentic=False,
                        retrieve=lambda _q, _k, cached=chunks: cached,
                    )
                    detail["generation"] = {
                        "answer": result.answer,
                        "status": result.status,
                        "timings": result.timings,
                        "error": result.error,
                        "cites_valid": sum(bool(c.get("valid")) for c in (result.citations.citations if result.citations else [])),
                    }
                except Exception as exc:
                    detail["generation"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            detailed.append(detail)
            print(f"{row['case_id']} | {method_name:<9} | Hit@{args.top_k}={row['hit_at_k']} | "
                  f"MRR={row['reciprocal_rank']:.3f} | {elapsed:.1f} ms" + (f" | ERROR={error}" if error else ""))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_by_method(rows)
    report = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_indice": store.index_version(),
        "space": args.space or "default",
        "modelo_embeddings": settings.EMBEDDING_MODEL,
        "top_k": args.top_k,
        "numero_preguntas": len(cases),
        "definicion_relevancia": "source igual + page opcional + al menos uno de expected_terms (si se especifica)",
        "advertencia": "Hit@k y MRR verifican criterios documentales; no miden exactitud factual de la respuesta.",
        "metrics": summary,
        "cases": detailed,
    }
    json_path = output_dir / "evaluacion_rag.json"
    csv_path = output_dir / "evaluacion_rag.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nRESUMEN DE EVALUACIÓN")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nEvidencias creadas: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
