import json
import re

import requests

from OntologyCreation import DATA_PATH
from OntologyCreation.nlp import is_mentioned

_MAX_SENTENCES = 5


def _truncate(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:_MAX_SENTENCES])


def get_dbpedia_data(concept: str, limit: int = -1):

    limit_clause = f"LIMIT {limit}" if limit != -1 else ""
    start_concept = concept
    query = """
    SELECT DISTINCT ?subclass ?parent WHERE {
        ?subclass rdfs:subClassOf ?parent .
        ?parent rdfs:subClassOf* <http://dbpedia.org/ontology/CONCEPT> .
    }
    LIMIT_CLAUSE
    """.replace("CONCEPT", start_concept).replace("LIMIT_CLAUSE", limit_clause)

    r = requests.get("https://dbpedia.org/sparql", params={"query": query, "format": "json"})

    results = r.json()

    headers = {"User-Agent": "MyOntologyProject/1.0"}
    data_formatted: dict[str, dict] = {}
    data_formatted["classes"] = {}

    unique_items = list(set([x["subclass"]["value"] for x in results["results"]["bindings"]]))

    start_uri = "http://dbpedia.org/ontology/CONCEPT".replace("CONCEPT", start_concept)
    unique_items += [start_uri]
    wiki_found_items = []

    for item in unique_items:
        short_name = item.split("/")[-1]
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{short_name}", headers=headers
        )

        if response.status_code != 200:
            continue

        body = response.json()
        if body.get("type") == "disambiguation":
            continue
        summary = body.get("extract", "")
        if not summary:
            continue

        wiki_found_items.append(item)
        data_formatted["classes"][short_name] = {
            "uri": item,
            "summary": _truncate(summary),
            "origin": "wikipedia",
        }

    hierarchy: dict[str, list] = {}
    for x in results["results"]["bindings"]:
        child_uri = x["subclass"]["value"]
        parent_uri = x["parent"]["value"]
        if child_uri not in wiki_found_items or parent_uri not in wiki_found_items:
            continue
        child = child_uri.split("/")[-1]
        parent = parent_uri.split("/")[-1]
        child_summary = data_formatted["classes"].get(child, {}).get("summary") or ""
        parent_summary = data_formatted["classes"].get(parent, {}).get("summary") or ""
        mentioned = is_mentioned(child, parent_summary, parent, child_summary)
        hierarchy.setdefault(child, []).append({"parent": parent, "mentioned": mentioned})
    data_formatted["hierarchy"] = hierarchy

    out_dir = DATA_PATH / "dbpedia"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{start_concept}.json", "w") as f:
        json.dump(data_formatted, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("concept")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    get_dbpedia_data(args.concept, args.limit)
