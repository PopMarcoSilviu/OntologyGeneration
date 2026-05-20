import json
import logging
import re
import time
from collections import deque

import requests
from nltk.corpus import wordnet

from OntologyCreation import DATA_PATH
from OntologyCreation.nlp import is_mentioned

_HEADERS = {"User-Agent": "OntologyCreation/1.0"}
_MAX_SENTENCES = 5

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
_log = logging.getLogger(__name__)


def _wiki_summary(name: str) -> str | None:
    """Fetch Wikipedia summary for a name, return up to 5 sentences or None."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}"
    while True:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        if resp.status_code == 429:
            time.sleep(1)
            continue
        if resp.status_code != 200:
            _log.warning("%s — HTTP %d", name, resp.status_code)
            return None
        break

    data = resp.json()
    if data.get("type") == "disambiguation":
        _log.warning("%s — disambiguation page", name)
        return None
    extract = data.get("extract", "")
    if not extract:
        _log.warning("%s — empty extract", name)
        return None
    _log.info("%s — ok", name)
    sentences = re.split(r"(?<=[.!?])\s+", extract.strip())
    return " ".join(sentences[:_MAX_SENTENCES])


def get_wordnet_data(concept: str, depth: int = -1):
    synsets = wordnet.synsets(concept, pos=wordnet.NOUN)
    if not synsets:
        raise ValueError(f"No WordNet noun synsets found for '{concept}'")
    root = synsets[0]

    # BFS to collect all hyponyms up to optional depth
    visited: dict = {}  # synset -> depth
    queue: deque = deque([(root, 0)])
    while queue:
        syn, d = queue.popleft()
        if syn in visited:
            continue
        if depth != -1 and d > depth:
            continue
        visited[syn] = d
        for hypo in syn.hyponyms():
            if hypo not in visited:
                queue.append((hypo, d + 1))

    # Assign names — use first lemma, fall back to synset key on collision
    name_map: dict = {}
    used_names: set[str] = set()
    for syn in visited:
        name = syn.lemmas()[0].name().replace("_", " ")
        if name in used_names:
            name = f"{name} ({syn.name()})"
        name_map[syn] = name
        used_names.add(name)

    # Build classes — prefer Wikipedia summary, fall back to WordNet definition
    classes = {}
    wiki_found = 0
    for syn, name in name_map.items():
        text = _wiki_summary(name)
        if text:
            wiki_found += 1
            origin = "wikipedia"
        else:
            definition = syn.definition()
            if syn.examples():
                definition += " " + " ".join(syn.examples())
            text = f"{name} definition is: {definition}"
            origin = "wordnet"
            _log.warning("%s — falling back to WordNet definition", name)
        classes[name] = {"uri": syn.name(), "summary": text, "origin": origin}

    # Build hierarchy (supports multiple parents via WordNet multiple inheritance)
    hierarchy: dict = {}
    for syn, child_name in name_map.items():
        child_summary = classes[child_name]["summary"]
        for hypernym in syn.hypernyms():
            if hypernym not in name_map:
                continue
            parent_name = name_map[hypernym]
            parent_summary = classes[parent_name]["summary"]
            mentioned = is_mentioned(child_name, parent_summary, parent_name, child_summary)
            hierarchy.setdefault(child_name, []).append(
                {"parent": parent_name, "mentioned": mentioned}
            )

    out_dir = DATA_PATH / "wordnet"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{concept}_{depth}.json", "w") as f:
        json.dump({"classes": classes, "hierarchy": hierarchy}, f, indent=4, ensure_ascii=False)

    print(
        f"Wrote {len(classes)} classes, {sum(len(v) for v in hierarchy.values())} edges "
        f"({wiki_found} wiki summaries, {len(classes) - wiki_found} wordnet fallbacks)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("concept")
    parser.add_argument("--depth", type=int, default=-1)
    args = parser.parse_args()

    get_wordnet_data(args.concept, args.depth)
