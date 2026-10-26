# RAG Knowledge Pipeline v2.1
## Implementación aplicada — Universidad Amazónica de Pando

### Objetivo
Implementación y evaluación experimental de un sistema RAG
basado en Python, ChromaDB, Sentence Transformers y Ollama.

### Mejoras incorporadas
- Diagnóstico automático del corpus documental.
- Inventario de documentos, fragmentos y metadatos.
- Evaluación comparativa de recuperación:
  Semantic, BM25, Hybrid y RRF.
- Métricas Hit@k, MRR y latencia.
- Exportación de resultados a CSV y JSON.
- Pruebas unitarias adicionales.

### Requisitos
- Linux, Ubuntu o WSL2
- Python 3.10 a 3.12
- Git
- Conexión a Internet para descargar modelos
- Ollama opcional para generación de respuestas

### 1. Clonar

git clone https://github.com/franzmoscoso/rag-knowledge-pipeline-v2.git
cd rag-knowledge-pipeline-v2

### 2. Preparar entorno

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

cp .env.example .env

### 3. Descargar documento de estudio

mkdir -p data/raw

curl -fL \
  https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf \
  -o data/raw/dataflow.pdf

### 4. Ejecutar pruebas unitarias

python -m pytest -q

### 5. Ingestar documentos

python cli.py ingest --source data/raw --reset

### 6. Diagnóstico

python scripts/diagnostico_rag.py

### 7. Evaluación comparativa

python scripts/evaluar_rag.py \
  --cases data/casos_evaluacion.json \
  --top-k 4

### 8. Consultas sin Ollama

python cli.py ask \
  "¿Qué diferencia existe entre event time y processing time?" \
  --no-generate

### 9. Generación de respuestas

Requiere instalar Ollama y descargar el modelo configurado
en .env.

### Resultados

Los experimentos generan archivos dentro de:

evidencias/

Incluyendo:
- diagnostico_corpus.json
- evaluacion_rag.csv
- evaluacion_rag.json

### Nota

Los resultados experimentales no deben interpretarse como
validación automática de la exactitud factual del LLM.

El corpus, los modelos y la configuración pueden influir
sobre las métricas y las respuestas.

### Compatibilidad

Las nuevas funcionalidades son aditivas.
No se eliminan las capacidades originales del proyecto.

### Documentación adicional

Consultar MEJORAS_V2_1.md y GUIA_EJECUCION_GPD.md.
