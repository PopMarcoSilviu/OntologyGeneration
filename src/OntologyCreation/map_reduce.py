import asyncio
import logging

import mlflow
import networkx as nx
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from OntologyCreation.models import (
    ClassesOnly,
    HierarchyOnly,
    Ontology,
    merge_classes,
    merge_hierarchy,
)
from OntologyCreation.nlp import term_variants
from OntologyCreation.utils import chunk_by_words, word_count

_log = logging.getLogger(__name__)


def _build_cooccurrence_graph(
    items: list[tuple[str, str]],
    all_classes: set[str],
) -> nx.Graph:
    """Paragraph nodes; edge between two paragraphs if they mention the same class."""
    G: nx.Graph = nx.Graph()
    for name, summary in items:
        G.add_node(name, summary=summary, words=word_count(summary))

    class_to_paras: dict[str, list[str]] = {}
    for name, summary in items:
        summary_lower = summary.lower()
        for cls in all_classes:
            if any(v in summary_lower for v in term_variants(cls)):
                class_to_paras.setdefault(cls, []).append(name)

    for paras in class_to_paras.values():
        for i in range(len(paras)):
            for j in range(i + 1, len(paras)):
                u, v = paras[i], paras[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G


def _split_component(G: nx.Graph, nodes: set[str], max_words: int) -> list[set[str]]:
    """Recursively bisect a component until every piece fits within max_words."""
    total = sum(G.nodes[n]["words"] for n in nodes)
    if total <= max_words or len(nodes) < 2:
        return [nodes]

    sub = G.subgraph(nodes).copy()

    components = list(nx.connected_components(sub))
    if len(components) > 1:
        result = []
        for comp in components:
            result.extend(_split_component(G, set(comp), max_words))
        return result

    try:
        _, (left, right) = nx.stoer_wagner(sub)
    except Exception:
        node_list = list(nodes)
        mid = len(node_list) // 2
        left, right = node_list[:mid], node_list[mid:]

    return _split_component(G, set(left), max_words) + _split_component(G, set(right), max_words)


def _classes_in_component(items: list[tuple[str, str]], all_classes: set[str]) -> set[str]:
    """Return only the classes from all_classes that are mentioned in these paragraphs."""
    mentioned: set[str] = set()
    for _, summary in items:
        summary_lower = summary.lower()
        for cls in all_classes:
            if any(v in summary_lower for v in term_variants(cls)):
                mentioned.add(cls)
    return mentioned


async def _extract_classes(
    agent: Agent[None, ClassesOnly],
    chunk: list[tuple[str, str]],
    settings: ModelSettings,
) -> ClassesOnly:
    """Run pass-1 class extraction on a single chunk of (name, summary) pairs."""
    text = "\n\n=====\n\n".join(summary for _, summary in chunk)
    result = await agent.run(text, model_settings=settings)
    return result.output


async def _extract_hierarchy(
    agent: Agent[None, HierarchyOnly],
    component_items: list[tuple[str, str]],
    component_classes: set[str],
    settings: ModelSettings,
) -> HierarchyOnly:
    """Run pass-2 hierarchy extraction for one component, filtering to component_classes only."""
    classes_list = "\n".join(f"- {c}" for c in sorted(component_classes))
    text = "\n\n=====\n\n".join(summary for _, summary in component_items)
    message = f"Known classes:\n{classes_list}\n\n=====\n\n{text}"
    result = await agent.run(message, model_settings=settings)
    hier = result.output.subclass_of
    # keep only edges where both child and parent are known classes for this component
    known_lower = {c.lower() for c in component_classes}
    filtered = {
        child: [p for p in parents if p.lower() in known_lower]
        for child, parents in hier.items()
        if child.lower() in known_lower
    }
    return HierarchyOnly(subclass_of={k: v for k, v in filtered.items() if v})


async def run_map_reduce(
    node_model: str,
    edge_model: str,
    node_prompt_name: str,
    edge_prompt_name: str,
    items: list[tuple[str, str]],
    concept: str,
    cfg: dict,
) -> Ontology:
    """Two-pass map-reduce ontology extraction.

    Pass 1 extracts class names per word-count chunk in parallel.
    Pass 2 extracts subclass relationships per co-occurrence component in parallel,
    using only the classes known to each component.

    Args:
        node_model: Model string for pass-1 class extraction.
        edge_model: Model string for pass-2 hierarchy extraction.
        node_prompt_name: MLflow prompt registry name for pass-1.
        edge_prompt_name: MLflow prompt registry name for pass-2.
        items: List of (name, summary) pairs to process.
        concept: Top-level ontology concept (e.g. "Monkey").
        cfg: Config dict with keys: max_words, temperature, max_tokens.

    Returns:
        Merged Ontology with deduplicated classes and combined hierarchy.
    """
    max_words = cfg.get("max_words", 10000)
    settings = ModelSettings(
        temperature=cfg.get("temperature", 1.0),
        max_tokens=cfg.get("max_tokens", 8192),
        extra_body={"thinking": {"type": "disabled"}},
    )

    # ── Pass 1: extract classes per chunk ────────────────────────────────────
    chunks = chunk_by_words(items, max_words)
    _log.info("Pass 1: %d chunks from %d items", len(chunks), len(items))

    node_prompt = mlflow.genai.load_prompt(node_prompt_name)  # pyright: ignore
    node_agent = Agent(
        node_model,
        output_type=ClassesOnly,
        instructions=str(node_prompt.format(concept=concept)),
        retries=2,
    )

    p1_results: list[ClassesOnly] = list(
        await asyncio.gather(*[_extract_classes(node_agent, chunk, settings) for chunk in chunks])
    )
    all_classes = set(merge_classes(p1_results).classes)
    _log.info("Pass 1 done: %d classes", len(all_classes))

    # ── Build co-occurrence graph and split large components ──────────────────
    G = _build_cooccurrence_graph(items, all_classes)
    components: list[set[str]] = []
    for comp in nx.connected_components(G):
        components.extend(_split_component(G, set(comp), max_words))
    _log.info("%d components after splitting", len(components))

    # ── Pass 2: extract hierarchy per component ───────────────────────────────
    edge_prompt = mlflow.genai.load_prompt(edge_prompt_name)  # pyright: ignore
    edge_agent = Agent(
        edge_model,
        output_type=HierarchyOnly,
        instructions=str(edge_prompt.format(concept=concept)),
        retries=1,
    )

    component_items_list = [[(n, G.nodes[n]["summary"]) for n in comp] for comp in components]

    p2_results: list[HierarchyOnly] = list(
        await asyncio.gather(
            *[
                _extract_hierarchy(
                    edge_agent,
                    comp_items,
                    _classes_in_component(comp_items, all_classes),
                    settings,
                )
                for comp_items in component_items_list
            ]
        )
    )

    merged = merge_classes(p1_results) | merge_hierarchy(p2_results)
    _log.info(
        "Merged: %d classes, %d hierarchy entries",
        len(merged.classes),
        sum(len(v) for v in merged.subclass_of.values()),
    )
    return merged
