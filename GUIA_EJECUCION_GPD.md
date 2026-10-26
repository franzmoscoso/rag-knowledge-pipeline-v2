# Guía de ejecución del RAG v2.1 en GPD + WSL2 (16 GB RAM)

## Propósito
Ejecución aplicada del pipeline RAG: conserva la funcionalidad original, adiciona
el diagnóstico del corpus y la evaluación de métodos de recuperación.

## Sistema
- Windows 10/11, WSL2 Ubuntu 22.04/24.04 (Python 3.10–3.12).
- Trabajo dentro del sistema de archivos Linux: `~/projects/`, **no** `/mnt/c`.
- Modelos: embeddings originales y Ollama `qwen2.5:1.5b` para reducir consumo de RAM.
- Sin Docker ni GPU obligatoria.

## Arranque

```bash
cd ~/projects/rag-knowledge-pipeline-v2
source .venv/bin/activate
python cli.py --help
bash ~/rag-gpd-starter/01_validar.sh
bash ~/rag-gpd-starter/02_ingesta.sh
bash ~/rag-gpd-starter/03_evaluar.sh
bash ~/rag-gpd-starter/activar_ollama.sh  # opcional, para respuestas generadas
bash ~/rag-gpd-starter/04_preguntas.sh
```

## Interfaz
```bash
cd ~/projects/rag-knowledge-pipeline-v2
source .venv/bin/activate
python -m streamlit run app/streamlit_app.py --server.address 0.0.0.0
```
Abrir en Windows `http://localhost:8501`.

## VS Code
Instalar VS Code y extensión WSL en Windows. En terminal WSL dentro del proyecto: `code .`.
Capturas de ventana VS Code + terminal integrada + salida real del comando.

## Métodos nuevos

```bash
python scripts/diagnostico_rag.py
python scripts/evaluar_rag.py --cases data/casos_evaluacion.json --top-k 4 --rrf
python scripts/evaluar_rag.py --cases data/casos_evaluacion.json --top-k 4 --generate
```

El experimento compara búsqueda semántica, BM25, híbrida y (opcional) RRF.
Hit@4/MRR se basan en el documento y términos esperados, **no** equivalen a
exactitud factual; contrastar cada respuesta con el PDF manualmente.

## Fuente de consulta

Artículo académico: Tyler Akidau et al., _The Dataflow Model: A Practical Approach to
Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing_.
Descargar de: https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf
Guardarlo como: `data/raw/dataflow.pdf`. Registrar título, autores y enlace en Word.

## Entrega

```bash
bash ~/rag-gpd-starter/05_entrega.sh
```

El informe Word base está **fuera** del repositorio, en:
`~/projects/rag-gpd-informe/Informe_base_original.docx`.
Para abrir la carpeta desde Windows: `explorer.exe .` en WSL o ruta `\\wsl$\Ubuntu\home\<usuario>\projects`.
No publicar informes ni capturas con datos personales en un repositorio público.

## Publicar mejoras cuando terminen de validar

```bash
cd ~/projects/rag-knowledge-pipeline-v2
git status --short
git add scripts/diagnostico_rag.py scripts/evaluar_rag.py \
  tests/test_mejoras_rag.py data/casos_evaluacion.json MEJORAS_V2_1.md GUIA_EJECUCION_GPD.md
git diff --cached --stat
git commit -m "feat: add corpus diagnostic and RAG retrieval benchmark"
git push origin main
```

Requiere credenciales de GitHub válidas. No ejecutar `git add .`: evitar subir
`.env`, PDFs e índices binarios de experimentación.
