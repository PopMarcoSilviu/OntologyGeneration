# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavior
- Ask clarifying questions before starting any non-trivial task
- Push back if an approach seems overly complex or has a better alternative
- Ask "why" when a design decision isn’t obvious
- Prefer the simplest solution that works
- Flag potential issues before implementing, not after
- Don’t assume — confirm ambiguous requirements first

## Project Overview

OntologyCreation uses Pydantic AI agents to automatically extract OWL ontology TBox definitions (class hierarchies) from unstructured text. The pipeline fetches data from DBpedia or WordNet/Wikipedia, feeds Wikipedia summaries to a Claude-powered agent, then evaluates extraction quality against the ground truth hierarchy. The evaluation framework is at least as important as ontology generation.

## Environment Setup

Manage dependencies with `uv`:

```bash
uv sync            # install all dependencies
uv add <package>   # add a new dependency
```

## Common Commands

```bash
# Fetch ontology data (taskipy tasks — use uv run)
uv run task get-dbpedia Animal           # DBpedia source
uv run task get-dbpedia Animal --limit 50
uv run task get-wordnet cat --depth 3    # WordNet source

# Run the extraction agent and evaluate
uv run task test                         # uses configs/baseline.yaml
python -m OntologyCreation.test --config configs/baseline.yaml

# Explore data and results
uv run task explore                      # Streamlit UI

# Lint, format, type check
ruff check . --fix && ruff format .
mypy src/

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Architecture

### Data Sources

Data lives in `data/<source>/<Concept>.json`, where source is `dbpedia` or `wordnet`. Both produce the same JSON schema:
- `classes`: `{name: {uri, wikipedia_summary}}` — Wikipedia summary (up to 5 sentences), or for WordNet fallback `"<name> definition is: <wordnet_def>"`
- `hierarchy`: `{child: [{parent, mentioned}]}` — supports multiple parents (WordNet multiple inheritance); `mentioned` is computed by `nlp.is_mentioned`

`get_dbpedia_data.py` queries DBpedia SPARQL for subclass hierarchy then fetches Wikipedia for each URI.
`get_wordnet_data.py` does BFS over WordNet hyponyms from a root synset, fetches Wikipedia by lemma name with retry-on-429, falls back to WordNet definition.

### Extraction + Evaluation Loop (`test.py`)

`run(cfg)` is the main entry point — it:
1. Loads `data/<source>/<ontology>.json` based on `configs/baseline.yaml`
2. Shuffles and concatenates all `wikipedia_summary` values into one text block
3. Passes it to a Pydantic AI `Agent` with output type `OntologyClasses`
4. Calls `evaluate()` which returns `(metrics_dict, details_dict)`
5. Logs metrics + `details_dict` as `diff/results.json` artifact to MLflow

`OntologyClasses` (`test.py:35`) — `classes: list[str]`, `subclass_of: dict[str, list[str]]` (child → list of parents). This is the structured output contract with the model.

`evaluate()` (`test.py:118`) — case-insensitive; uses `nlp.build_lemma_map` to fuzzy-match extracted names to ground truth when `strict=False`. Returns precision/recall for classes and hierarchy, plus the Venn breakdown (in_original / extra / missing) used for the diff artifact.

### NLP Utilities (`nlp.py`)

- `term_variants(term)` — returns term + lemma form + WordNet synset lemma names; used by `is_mentioned`
- `is_mentioned(term_a, summary_b, term_b, summary_a)` — true if any variant of either term appears in the other’s summary; used to flag hierarchy edges that have Wikipedia cross-reference evidence
- `build_lemma_map(extracted, real)` — maps extracted class names to ground-truth names via lemma/synset matching; used in `evaluate()` for non-strict mode

### Prompt Management

System prompt lives in the MLflow prompt registry under `ontology_extraction`. `run()` always loads `@latest`. Register a new version in MLflow to iterate on prompts without code changes.

### Visualization (`explore.py`)

Streamlit app with two tabs:
- **Class Cards** — shows parents/children, highlights term variants in summary text, recomputes `is_mentioned` live
- **Graph** — fetches `diff/results.json` artifact from a selected MLflow run, renders ground truth vs. extracted side-by-side using pyvis; green nodes/edges = found, red = missing, orange = extra

### Config (`configs/baseline.yaml`)

Keys: `ontology`, `source` (dbpedia|wordnet), `temperature`, `prompt`, `shuffle_input_test`, `strict_match`. `MODEL` is read from the environment (`.env`).

## Scalability Note

For large ontologies the single-call approach hits output token limits. The planned fix is a two-pass map-reduce: pass 1 extracts class names per chunk (small output), pass 2 extracts hierarchy per chunk given the full deduplicated class list as context (avoids cross-chunk parent-assignment errors).

## Code Style

Line length is 100 characters (Ruff). Enabled rules: `E, F, W, I, UP, B`.
