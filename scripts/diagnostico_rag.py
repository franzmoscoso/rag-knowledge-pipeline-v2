"""Diagnóstico reproducible del corpus RAG. Solo lectura."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def summarize_records(records: list[dict]) -> dict:
    sizes = [len(r.get("text") or "") for r in records]
    pages = sum(r.get("page") is not None for r in records)
    return {
        "fragmentos": len(records),
        "fragmentos_con_pagina": pages,
        "caracteres_por_fragmento": {
            "minimo": min(sizes, default=0),
            "promedio": round(statistics.mean(sizes), 1) if sizes else 0,
            "maximo": max(sizes, default=0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Informe de la base RAG; no modifica documentos ni índices.")
    parser.add_argument("--space", default=None, help="Espacio de conocimiento, si corresponde")
    parser.add_argument("--output", default="evidencias/diagnostico_corpus.json")
    args = parser.parse_args()

    from config.settings import settings
    from rag.vector_store import VectorStore

    store = VectorStore(space=args.space)
    documents = store.list_sources()
    counts = summarize_records(store.get_all())
    report = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "espacio": args.space or "default",
        "coleccion": store.collection.name,
        "version_indice": store.index_version(),
        "modelo_embeddings": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "total_documentos": len(documents),
        "total_fragmentos": store.count(),
        "resumen": counts,
        "fuentes": documents,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nArchivo generado: {output}")
    if report["total_fragmentos"] == 0:
        print("ATENCIÓN: índice vacío; ejecute cli.py ingest antes de evaluar.")


if __name__ == "__main__":
    main()
