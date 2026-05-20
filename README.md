# OntologyCreation

Using agents in order to create an OWL ontology (mostly TBox) from unstructured or semi-structured text.

## Setup

```bash
# Install uv (if not already installed)
pip install uv

# Install dependencies
uv sync

# Copy env template and fill in your API keys
cp .env.example .env
```

Required env vars in `.env`:
- `ANTHROPIC_API_KEY` — Anthropic API key
- `DEEPSEEK_API_KEY` — DeepSeek API key
- `MODEL_FAST` — fast model (e.g. `deepseek:deepseek-chat`)
- `MODEL_SMART` — smart model (e.g. `claude-haiku-4-5-20251001`)
- `MODEL` — default model for single-call mode

## Fetch data

```bash
uv run task get-wordnet monkey --depth 3    # WordNet source
uv run task get-dbpedia Animal              # DBpedia source
```

Data is saved to `data/<source>/<Concept>.json` (not committed — fetch locally).

## Run

```bash
uv run task test                            # uses configs/baseline.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db
```