# Extensión aplicada RAG v2.1 — diagnóstico y evaluación

## Objetivo

Agregar **diagnóstico de corpus** y **evaluación reproducible** a la versión existente de RAG. Es una extensión **aditiva**: no sustituye `cli.py`, ningún módulo del pipeline, la interfaz Streamlit, los tests originales, ni sus configuraciones. No agrega dependencias.

## Archivos nuevos

- `scripts/diagnostico_rag.py`: inventario de fuentes, fragmentos, páginas, configuración e índice.
- `scripts/evaluar_rag.py`: compara recuperación semántica, BM25 e híbrida (RRF opcional), mide Hit@k, MRR y latencias; exporta JSON y CSV, más la evidencia de los fragmentos recuperados. `--generate` registra opcionalmente la respuesta de Ollama para el método híbrido.
- `data/casos_evaluacion.json`: dos preguntas de ejemplo sobre el artículo Dataflow. **Ajustar `expected_source` y los criterios de referencia a la fuente realmente utilizada.**
- `tests/test_mejoras_rag.py`: pruebas de criterios, métricas y estadísticas sin servicios externos.

## Preparación

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
python -m unittest discover -s tests -p 'test_mejoras_rag.py' -v
python -m pytest -q
```

Descargar el PDF para el caso de estudio desde https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf y guardarlo como `data/raw/dataflow.pdf`. Se trata de una fuente de consulta propuesta; el equipo debe verificar sus permisos, la extracción y la pertinencia de las preguntas.

```bash
python cli.py ingest --source data/raw --reset
python scripts/diagnostico_rag.py
python scripts/evaluar_rag.py --cases data/casos_evaluacion.json --top-k 4
```

**Opcional**, si Ollama está instalado y se descargó el modelo especificado en `.env`:

```bash
python scripts/evaluar_rag.py --cases data/casos_evaluacion.json --top-k 4 --generate
```

Los resultados quedan en `evidencias/diagnostico_corpus.json`, `evidencias/evaluacion_rag.json` y `evidencias/evaluacion_rag.csv`. Conservar capturas auténticas de la terminal integrada de VS Code y de las respuestas.

## Interpretación

- **Hit@k**: proporción de preguntas para las que al menos un fragmento entre los primeros *k* cumple los criterios de referencia (`expected_source`, página si se configuró y al menos uno de los términos previstos si se configuraron).
- **MRR**: promedio del recíproco de la posición del primer fragmento que coincide con los criterios.
- **Latencia**: tiempo de recuperación en la ejecución concreta; el primer uso puede incluir cargas de modelo. No comparar directamente puntajes de similitud con scores BM25/RRF.
- **Precaución**: las métricas indican recuperación de evidencia según etiquetas del equipo, no prueban que la respuesta sea factual ni que las citas sean correctas. Se deben revisar manualmente ambas respuestas contra el PDF.

## Material para el informe

1. **Cambios propios**: dos scripts, casos JSON y pruebas unitarias. Mostrar el `git diff --stat` y el resultado de tests.
2. **Diagnóstico**: modelo, tamaño de chunk, número de fuentes, páginas y registros recuperables.
3. **Experimento**: preguntas, criterios de relevancia establecidos previamente, tabla Hit@k/MRR/latencia por método, fragmentos y capturas de respuesta.
4. **Limitaciones**: conjunto de solo dos preguntas, juicio parcialmente léxico, efecto de la configuración y ausencia de evaluación humana exhaustiva.
5. **Reproducibilidad**: indicar SO, Python, versión del proyecto, versión de Ollama/modelo, hardware y fecha de la prueba.

## Comprobación de que las funciones existentes se conservan

```bash
python cli.py --help
python cli.py ask --help
python -m pytest -q
```

Esta extensión no modifica el repositorio remoto por sí sola: copiar los archivos, probar y luego usar `git add`, `git commit` y `git push`.
