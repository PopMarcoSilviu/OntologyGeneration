import argparse
import json
import logging
import os
import random

import mlflow
import yaml
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

from OntologyCreation import DATA_PATH, MLFLOW_URI
from OntologyCreation.map_reduce import run_map_reduce
from OntologyCreation.models import Ontology
from OntologyCreation.nlp import build_lemma_map
from OntologyCreation.utils import format_diff, word_count

_log = logging.getLogger(__name__)

mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]
load_dotenv()
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("ontology-extraction")


def evaluate(
    extracted: Ontology,
    ground_truth: Ontology,
    mentionable_pairs: set,
    strict: bool = False,
):
    """Compute precision/recall metrics and Venn breakdown against ground truth.

    Args:
        extracted: Ontology produced by the extraction pipeline.
        ground_truth: Reference ontology from the data source.
        mentionable_pairs: Hierarchy pairs that have Wikipedia cross-reference evidence.
        strict: If True, skip lemma fuzzy-matching and require exact name matches.

    Returns:
        Tuple of (metrics dict, details dict). metrics has class/hierarchy
        precision and recall; details has in_original/extra/missing breakdowns.
    """
    extracted_classes = {c.lower() for c in extracted.classes}
    real_classes_lower = {c.lower() for c in ground_truth.classes}
    real_pairs = {
        (k.lower(), p.lower()) for k, parents in ground_truth.subclass_of.items() for p in parents
    }

    lemma_map = {} if strict else build_lemma_map(extracted_classes, real_classes_lower)

    def _annotate(name: str) -> str:
        mapped = lemma_map.get(name, name)
        return f"{mapped} ({name})" if mapped != name else name

    normalized_extracted = {lemma_map.get(c, c) for c in extracted_classes}
    extracted_pairs = {
        (k.lower(), p.lower()) for k, parents in extracted.subclass_of.items() for p in parents
    }
    normalized_pairs = {(lemma_map.get(c, c), lemma_map.get(p, p)) for c, p in extracted_pairs}

    class_tp = len(normalized_extracted & real_classes_lower)
    class_precision = class_tp / len(normalized_extracted) if normalized_extracted else 0
    class_recall = class_tp / len(real_classes_lower) if real_classes_lower else 0

    hier_correct = len(normalized_pairs & real_pairs)
    hier_precision = hier_correct / len(normalized_pairs) if normalized_pairs else 0
    hier_recall = hier_correct / len(real_pairs) if real_pairs else 0

    hier_mentionable_correct = len(normalized_pairs & mentionable_pairs)
    hier_recall_mentionable = (
        hier_mentionable_correct / len(mentionable_pairs) if mentionable_pairs else 0
    )

    metrics = {
        "class_precision": class_precision,
        "class_recall": class_recall,
        "hierarchy_precision": hier_precision,
        "hierarchy_recall": hier_recall,
        "hierarchy_recall_mentionable": hier_recall_mentionable,
    }
    details = {
        "classes": {
            "in_original": sorted(
                _annotate(c) for c in extracted_classes if lemma_map.get(c, c) in real_classes_lower
            ),
            "extra": sorted(
                _annotate(c)
                for c in extracted_classes
                if lemma_map.get(c, c) not in real_classes_lower
            ),
            "missing": sorted(real_classes_lower - normalized_extracted),
        },
        "hierarchy": {
            "in_original": sorted(
                f"{_annotate(c)} -> {_annotate(p)}"
                for c, p in extracted_pairs
                if (lemma_map.get(c, c), lemma_map.get(p, p)) in real_pairs
            ),
            "extra": sorted(
                f"{_annotate(c)} -> {_annotate(p)}"
                for c, p in extracted_pairs
                if (lemma_map.get(c, c), lemma_map.get(p, p)) not in real_pairs
            ),
            "missing": sorted(f"{c} -> {p}" for c, p in real_pairs - normalized_pairs),
            "mentionable_found": sorted(
                f"{_annotate(c)} -> {_annotate(p)}"
                for c, p in extracted_pairs
                if (lemma_map.get(c, c), lemma_map.get(p, p)) in mentionable_pairs
            ),
            "mentionable_missing": sorted(
                f"{c} -> {p}" for c, p in mentionable_pairs - normalized_pairs
            ),
        },
    }
    return metrics, details


async def run(cfg: dict):
    """Load data, run extraction (single-call or map-reduce), evaluate, and log to MLflow.

    Args:
        cfg: Config dict loaded from a YAML file (see configs/baseline.yaml).
    """
    model = os.getenv("MODEL", "")

    with open(DATA_PATH / cfg.get("source", "dbpedia") / f"{cfg['ontology']}.json") as f:
        data = json.load(f)

    items = list(data["classes"].items())
    if cfg.get("shuffle_input_test", True):
        random.shuffle(items)

    ground_truth = Ontology(
        classes=list(data["classes"].keys()),
        subclass_of={c: [p["parent"] for p in parents] for c, parents in data["hierarchy"].items()},
    )
    mentionable_pairs = {
        (c.lower(), p["parent"].lower())
        for c, parents in data["hierarchy"].items()
        for p in parents
        if p["mentioned"]
    }

    use_map_reduce = "node_prompt" in cfg and "edge_prompt" in cfg
    total_words = sum(word_count(value["summary"]) for _, value in items)

    with mlflow.start_run() as active_run:
        print("run id:", active_run.info.run_id)
        mlflow.log_params(
            {k: v for k, v in cfg.items() if k not in ("prompt", "node_prompt", "edge_prompt")}
        )

        if use_map_reduce:
            _log.info("Using map-reduce (%d words total)", total_words)
            mlflow.log_param("mode", "map_reduce")
            node_model = os.environ[cfg["node_model"]]
            edge_model = os.environ[cfg["edge_model"]]
            mlflow.log_params({"node_model_name": node_model, "edge_model_name": edge_model})
            output = await run_map_reduce(
                node_model=node_model,
                edge_model=edge_model,
                node_prompt_name=cfg["node_prompt"],
                edge_prompt_name=cfg["edge_prompt"],
                items=[(name, v["summary"]) for name, v in items],
                concept=cfg["ontology"],
                cfg=cfg,
            )
        else:
            _log.info("Using single call (%d words total)", total_words)
            mlflow.log_param("mode", "single_call")
            prompt = mlflow.genai.load_prompt(cfg["prompt"])  # pyright: ignore[reportPrivateImportUsage]
            mlflow.log_param("prompt_version", prompt.version)
            agent = Agent(
                model,
                output_type=Ontology,
                instructions=str(prompt.format(concept=cfg["ontology"])),
            )
            result = await agent.run(
                "\n\n=====\n\n".join(v["summary"] for _, v in items),
                model_settings=AnthropicModelSettings(
                    temperature=cfg.get("temperature", 1.0),
                    max_tokens=cfg.get("max_output_tokens", 8192),
                ),
            )
            output = result.output
            print("tokens:", result.usage().total_tokens)

        metrics, details = evaluate(
            output,
            ground_truth,
            mentionable_pairs,
            strict=cfg.get("strict_match", False),
        )

        mlflow.log_metrics(metrics)
        mlflow.log_metrics(
            {
                "in_original_classes": len(details["classes"]["in_original"]),
                "extra_classes": len(details["classes"]["extra"]),
                "missing_classes": len(details["classes"]["missing"]),
            }
        )
        mlflow.log_dict(details, "diff/results.json")
        mlflow.log_text(format_diff(details), "diff/summary.txt")

        print(metrics)

    return output


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    asyncio.run(run(cfg))
