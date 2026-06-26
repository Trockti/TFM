# TFM Project

This repository contains the code and data used for the TFM project focused on lexical simplification, dictionary generation, definition extraction, evaluation, and model benchmarking.

## Project overview

The project combines several components:

- `application/` — web application, backend API, and frontend interface.
- `detection_module/` — detection and preprocessing logic.
- `generation_module/` — definition generation and model usage workflows.
- `dictionary_programs/` — dictionary extraction, normalization, and analysis utilities.
- `metrics_module/` — evaluation, readability, and metric computation.
- `data/` — benchmark datasets, extracted definitions, and dictionaries.
- `tests/` — validation and regression tests.

## Main goals

- Generate and refine definitions for Spanish terms.
- Evaluate model quality using linguistic and readability metrics.
- Support experiments with multiple models and datasets.
- Provide a usable application layer for testing and demonstration.

## Requirements

The project depends on Python packages listed in `requirements.txt`.

Recommended setup:

```bash
cd TFM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Useful folders

- `application/backend/` — backend services and API entry points.
- `application/frontend/` — Angular frontend.
- `detection_module/` — detection of botrrowings in a given text.
- `generation_module/` — generation of definitions using different models.
- `metrics_module/` — metric computation and experimental analysis.
- `dictionary_programs/` — programs used to generate all the data.
- `data/dictionaries/` — all the versions of the dictionaries generated.
- `data/test_datasets/` — all the versions of the test datasets generated.

## Generating Definitions

The `generation_module/use_models.py` script generates simplified definitions for terms using multiple language models.

**Available Models:**
- **llama** — meta-llama/Llama-3.2-3B-Instruct (Multilingual)
- **ministral** — mistralai/Ministral-3-8B-Instruct-2512 (Multilingual)
- **latxa** — HiTZ/Latxa-Qwen3-VL-8B-Instruct (Spanish/Catalan/Basque specialized)
- **rigochat** — IIC/RigoChat-7b-v2 (Spanish language model)

**Usage:**

```bash
cd generation_module
python use_models.py
```

The script:
1. Loads a glossary dataset from `../data/dictionaries/transformed_terms.json`
2. Applies prompt templates to generate simplified definitions
3. Runs inference with the configured models
4. Saves results in JSON format with structure: `{field: {id_term, term, original, adapted, context, simplified}}`
5. Outputs are organized in version folders (e.g., `experiments/results_v1/`, `experiments/results_v2/`, etc.)

**Configuration:**
- Modify the `prompt_templates` list in the `main()` function to customize the prompt.
- Change `input_file` to point to your dataset.
- Specify which models to use via the model loading configuration.
- Results are saved with model name as filename (e.g., `llama.json`, `ministral.json`).

**Output Structure:**
Each model generates a JSON file containing:
```json
{
  "field_name": {
    "id_term": "...",
    "term": "...",
    "original": "...",
    "adapted": "...",
    "context": "...",
    "simplified": "Generated definition here"
  }
}
```

## Computing Metrics

The `metrics_module/hulat_metrics.py` script evaluates the quality of generated definitions using multiple linguistic and readability metrics.

**Supported Metrics:**
- **BERTScore** — semantic similarity between texts
- **ReadabilityScores** — Fernández Huerta (ES) and Flesch Reading Ease (EN)
- **SARI** — measures level of simplification
- **ROUGE** — overlap-based similarity metrics
- **Sentence Transformers** — semantic similarity using transformers
- **MoverScore** — cross-lingual semantic evaluation
- **AlignScore** — Natural Language Inference-based alignment scoring
- **SummaC** — factuality evaluation
- **Hallucination Detection** — identifies generated content not in source

**Usage:**

```bash
cd metrics_module
python hulat_metrics.py --lang es --input_folder /path/to/models/results --out /path/to/output
```

**Parameters:**
- `--lang` — Language for evaluation: `es` (Spanish) or `en` (English) [required]
- `--input_folder` — Path to folder containing model JSON files, where filename = model name [required]
- `--out` — Path to output folder for results (JSON, CSV, and CodeCarbon emissions reports) [required]

**Input Format:**
The script expects JSON files in `input_folder` with the following structure:
```json
{
  "field_name": {
    "original": "original text",
    "simplified": "simplified definition",
    "reference": "reference definition"
  }
}
```

**Outputs:**
- **`metrics_{model}.json`** — Full metric results per model
- **`metrics_{model}.csv`** — Metric results in tabular format
- **`emissions_metricas_{model}.csv`** — Carbon emissions tracking


**Example:**
```bash
python hulat_metrics.py --lang es --input_folder ../generation_module/experiments/results_v1 --out ./experiments/results_v1
```

This will:
1. Read all JSON files from `experiments/results_v1/` (e.g., `llama.json`, `ministral.json`, etc.)
2. Compute metrics for each model
3. Generate comparison reports in `experiments/results_v1/`

alternatively, one can use the execute_metrics.sh script to automatically compute all the metrics, using:

```bash
./execute_metrics.sh
```
## Using the Application

The EASIER application provides a web-based interface for lexical simplification and borrowing detection in Spanish texts. It consists of a backend API and an Angular frontend.

### Backend Setup

**Prerequisites:**
- Python 3.8+
- CUDA-enabled GPU (recommended for faster inference)
- Hugging Face API token (for model access)

**Installation:**

```bash
cd application/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Configuration:**
Create a `.env` file in `application/backend/`:
```
HUGGINGFACE_TOKEN=your_hf_token_here
```

**Running the Backend:**

```bash
cd application/backend
./restart_container.sh
```


The API server will start on `http://localhost:5000`.


### Frontend Setup

**Prerequisites:**
- Node.js 12+ and npm

**Installation:**

```bash
cd application/frontend
npm install
```


### Complete Workflow

1. **Start the backend:**
   ```bash
   cd application/backend
   ./restart_container.sh
   ```

2. **Start the frontend:**
   ```bash
   cd application/frontend
   ng serve
   ```

3. **Access the application:**
   - Open `http://localhost:4200` in your browser
   - Enter Spanish text to analyze
   - The system will identify borrowings and complex words
   - Request definitions and simplifications through the API


## Notes

- Some scripts rely on local datasets stored under `data/`.
- Environment variables may be required by the application and model integrations, such as a Hugging Face API key.
- The project is intended for research, experimentation, and prototype development.
- GPU support requires CUDA-compatible PyTorch installation.

## Project Context
This Bachelor's Thesis was developed within the framework of the research project: **PID2023-148577OB-C21** *Human-Centered AI: User-Driven Adapted Language Models (HUMAN_AI)* Funded by: MICIU- AEI (10.13039/501100011033) - FEDER/European Union

