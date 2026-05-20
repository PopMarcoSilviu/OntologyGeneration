import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from mlflow.tracking import MlflowClient
from pyvis.network import Network

from OntologyCreation import MLFLOW_URI
from OntologyCreation.nlp import is_mentioned
from OntologyCreation.nlp import term_variants as _term_variants


def _parse_diff_name(s: str) -> str:
    """Extract lowercase DBpedia name from 'dbpedia_name (wiki_form)' or plain 'name'."""
    m = re.match(r"^(.+?)\s+\(.+\)$", s.strip())
    return (m.group(1) if m else s.strip()).lower()


def _parse_diff_edge(s: str) -> tuple[str, str]:
    """Parse 'child -> parent' diff string into (child_lower, parent_lower) DBpedia names."""
    child_str, parent_str = s.split(" -> ", 1)
    return _parse_diff_name(child_str), _parse_diff_name(parent_str)


DATA_PATH = Path(__file__).parent / "data"

st.set_page_config(page_title="Ontology Explorer", layout="wide")
st.title("Ontology Explorer")


@st.cache_data
def load_data(concept: str) -> dict:
    source, name = concept.split("/", 1)
    with open(DATA_PATH / source / f"{name}.json") as f:
        return json.load(f)


def available_concepts() -> list[str]:
    return sorted(f"{p.parent.name}/{p.stem}" for p in DATA_PATH.glob("*/*.json"))


def get_relations(data: dict, class_name: str) -> tuple[list[str], list[str]]:
    """Return ([parents], [children]) for a class."""
    h = data["hierarchy"]
    parents = [p["parent"] for p in h[class_name]] if class_name in h else []
    children = sorted(
        c for c, ps in h.items() if any(p["parent"].lower() == class_name.lower() for p in ps)
    )
    return parents, children


def highlight(text: str, terms: dict[str, str]) -> str:
    """Wrap each term occurrence in a coloured <mark>. terms = {word: hex_color}"""
    for term, color in terms.items():
        text = re.sub(
            re.escape(term),
            f'<mark style="background:{color};padding:1px 3px;border-radius:3px">{term}</mark>',
            text,
            flags=re.IGNORECASE,
        )
    return text


def class_card(data: dict, class_name: str) -> None:
    parents, children = get_relations(data, class_name)

    cols = st.columns([1, 1])
    with cols[0]:
        if parents:
            st.markdown("**Parents:** " + " ".join(f"`{p}`" for p in parents))
        else:
            st.markdown("*No parent (root)*")
    with cols[1]:
        entries = data["hierarchy"].get(class_name, [])
        class_summary = data["classes"].get(class_name, {}).get("summary", "")
        any_mentioned = any(
            is_mentioned(
                class_name,
                data["classes"].get(p["parent"], {}).get("summary", ""),
                p["parent"],
                class_summary,
            )
            for p in entries
        )
        if entries:
            if any_mentioned:
                st.markdown("🔗 **Mentioned** in wiki cross-reference")
            else:
                st.markdown("✗ Not mentioned in wiki cross-reference")

    if children:
        st.markdown("**Children:** " + " ".join(f"`{c}`" for c in children))

    summary = data["classes"][class_name]["summary"]
    color_map = {}
    for p in parents:
        for variant in _term_variants(p):
            color_map[variant] = "#ffd700"
    for child in children:
        for variant in _term_variants(child):
            color_map[variant] = "#90ee90"

    st.markdown("---")
    st.markdown(highlight(summary, color_map), unsafe_allow_html=True)

    if color_map:
        st.caption("🟡 parent   🟢 children")


# ── MLflow helpers ────────────────────────────────────────────────────────────


def get_runs(ontology: str) -> list[dict]:
    client = MlflowClient(tracking_uri=str(MLFLOW_URI))
    exp = client.get_experiment_by_name("ontology-extraction")
    if not exp:
        return []
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"params.ontology = '{ontology}'",
        order_by=["start_time DESC"],
    )
    return [
        {
            "run_id": r.info.run_id,
            "label": f"{r.info.run_id[:8]}  {datetime.fromtimestamp(r.info.start_time / 1000):%Y-%m-%d %H:%M}"
            f"  |  P={r.data.metrics.get('class_precision', 0):.2f}"
            f"  R={r.data.metrics.get('class_recall', 0):.2f}"
            f"  hier_R={r.data.metrics.get('hierarchy_recall', 0):.2f}",
            "metrics": r.data.metrics,
        }
        for r in runs
    ]


def load_diff(run_id: str) -> dict | None:
    client = MlflowClient(tracking_uri=str(MLFLOW_URI))
    try:
        local_dir = tempfile.mkdtemp()
        client.download_artifacts(run_id, "diff/results.json", local_dir)
        candidates = list(Path(local_dir).rglob("results.json"))
        if not candidates:
            return None
        with open(candidates[0]) as f:
            return json.load(f)
    except Exception:
        return None


# ── Graph helpers ─────────────────────────────────────────────────────────────


def _graph_options(
    spring_length: float, spring_constant: float, gravity: float, central_gravity: float
) -> str:
    return json.dumps(
        {
            "physics": {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": gravity,
                    "centralGravity": central_gravity,
                    "springLength": spring_length,
                    "springConstant": spring_constant,
                    "damping": 0.5,
                    "avoidOverlap": 0.5,
                },
                "stabilization": {"iterations": 300},
            },
            "edges": {
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                "smooth": {"type": "dynamic"},
                "color": "#aaaaaa",
                "width": 1,
            },
            "nodes": {
                "shape": "box",
                "font": {"size": 14, "color": "#111111"},
                "margin": 8,
            },
            "interaction": {"zoomView": True, "dragView": True, "hover": True},
        }
    )


# Node colours
_C_OK = "#a8e6cf"
_C_MISS = "#ffaaa5"
_C_EXTRA = "#ffd3b6"
_C_BORDER_OK = "#27ae60"
_C_BORDER_MISS = "#c0392b"
_C_BORDER_EXTRA = "#e67e22"


def _build_graph(
    data: dict,
    diff: dict,
    mode: str,
    spring_length: float = 60,
    spring_constant: float = 0.15,
    gravity: float = -25,
    central_gravity: float = 0.05,
) -> str:
    """Return pyvis HTML for the ontology graph."""
    net = Network(height="600px", width="100%", directed=True)
    net.set_options(_graph_options(spring_length, spring_constant, gravity, central_gravity))

    in_orig_classes = {_parse_diff_name(s) for s in diff["classes"]["in_original"]}
    missing_classes = {_parse_diff_name(s) for s in diff["classes"]["missing"]}
    in_orig_edges = {_parse_diff_edge(s) for s in diff["hierarchy"]["in_original"]}

    gt_mentioned: set[tuple[str, str]] = {
        (child.lower(), p["parent"].lower())
        for child, parents in data["hierarchy"].items()
        for p in parents
        if is_mentioned(
            child,
            data["classes"].get(p["parent"], {}).get("summary", ""),
            p["parent"],
            data["classes"].get(child, {}).get("summary", ""),
        )
    }

    # ── Ground truth nodes ────────────────────────────────────────────────────
    font = {"size": 14, "color": "#111111"}
    for cls in data["classes"]:
        key = cls.lower()
        if key in in_orig_classes:
            bg, border = _C_OK, _C_BORDER_OK
        elif key in missing_classes:
            bg, border = _C_MISS, _C_BORDER_MISS
        else:
            bg, border = "#f0f0f0", "#aaaaaa"
        net.add_node(
            cls, label=cls, shape="box", font=font, color={"background": bg, "border": border}
        )  # type: ignore[arg-type]

    # ── Extra nodes (right graph only) ────────────────────────────────────────
    if mode == "right":
        gt_lower = {c.lower() for c in data["classes"]}
        for raw in diff["classes"]["extra"]:
            key = _parse_diff_name(raw)
            if key not in gt_lower and key not in net.node_map:
                label = re.sub(r"\s*\(.*\)$", "", raw).strip()
                net.add_node(
                    key,
                    label=label,
                    shape="box",
                    font=font,  # type: ignore[arg-type]
                    color={"background": _C_EXTRA, "border": _C_BORDER_EXTRA},
                )

    # ── Ground truth edges ────────────────────────────────────────────────────
    for child, parents in data["hierarchy"].items():
        for p in parents:
            parent = p["parent"]
            edge_key = (child.lower(), parent.lower())
            edge_mentioned = edge_key in gt_mentioned

            if edge_mentioned:
                color, width = "#27ae60", 3  # green — mentioned
            elif edge_key in in_orig_edges:
                color, width = "#888888", 1.5  # dark grey — found, not mentioned
            else:
                color, width = "#cccccc", 1.5  # light grey — missing or unknown

            if parent in net.node_map and child in net.node_map:
                net.add_edge(
                    parent,
                    child,
                    color=color,
                    width=width,
                    title="★ mentioned" if edge_mentioned else "",
                )

    # ── Extra edges (right graph only) ───────────────────────────────────────
    if mode == "right":
        for raw in diff["hierarchy"]["extra"]:
            child_key, parent_key = _parse_diff_edge(raw)
            for key in (child_key, parent_key):
                if key not in net.node_map:
                    net.add_node(
                        key,
                        label=key,
                        shape="box",
                        font=font,  # type: ignore[arg-type]
                        color={"background": _C_EXTRA, "border": _C_BORDER_EXTRA},
                    )
            net.add_edge(
                parent_key, child_key, color="#e67e22", width=1.5, dashes=True, title="extra"
            )

    html = net.generate_html(notebook=False)
    inject = (
        "network.on('stabilizationIterationsDone',function(){"
        "network.setOptions({physics:{enabled:false}});"
        "network.fit({animation:false});});"
        "return network;"
    )
    return html.replace("return network;", inject)


# ── Concept selector ──────────────────────────────────────────────────────────
concepts = available_concepts()
concept = st.selectbox("Ontology", concepts)
data = load_data(concept)
all_classes = sorted(data["classes"].keys())

tab_cards, tab_graph = st.tabs(["Class Cards", "Graph"])

# ── Tab 1: Class Cards ────────────────────────────────────────────────────────
with tab_cards:
    left, right = st.columns(2)

    with left:
        cls_left = st.selectbox("Class", all_classes, key="left")
        class_card(data, cls_left)

    with right:
        default_right = all_classes[1] if len(all_classes) > 1 else all_classes[0]
        cls_right = st.selectbox(
            "Class", all_classes, index=all_classes.index(default_right), key="right"
        )
        class_card(data, cls_right)

# ── Tab 2: Graph ──────────────────────────────────────────────────────────────
with tab_graph:
    runs = get_runs(concept)
    if not runs:
        st.info("No MLflow runs found for this ontology.")
    else:
        run_labels = [r["label"] for r in runs]
        selected_idx = st.selectbox(
            "Run", range(len(run_labels)), format_func=lambda i: run_labels[i]
        )
        selected_run = runs[selected_idx]

        diff = load_diff(selected_run["run_id"])
        if diff is None:
            st.warning("No diff artifact found for this run (may predate artifact logging).")
        else:
            m = selected_run["metrics"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Class P", f"{m.get('class_precision', 0):.2f}")
            mc2.metric("Class R", f"{m.get('class_recall', 0):.2f}")
            mc3.metric("Hier P", f"{m.get('hierarchy_precision', 0):.2f}")
            mc4.metric("Hier R", f"{m.get('hierarchy_recall', 0):.2f}")
            mc5.metric("Hier R (ment.)", f"{m.get('hierarchy_recall_mentionable', 0):.2f}")

            st.caption("🟢 found   🔴 missed   🟠 extra   🟡 wiki-mentioned")

            with st.expander("Physics"):
                pc1, pc2, pc3, pc4 = st.columns(4)
                spring_length = pc1.slider("Spring length", 10, 300, 60)
                spring_constant = pc2.slider("Spring constant", 0.01, 0.5, 0.15, step=0.01)
                gravity = pc3.slider("Gravity", -200, -1, -25)
                central_gravity = pc4.slider("Central gravity", 0.001, 0.3, 0.05, step=0.005)

            physics = dict(
                spring_length=spring_length,
                spring_constant=spring_constant,
                gravity=gravity,
                central_gravity=central_gravity,
            )

            gcol_left, gcol_right = st.columns(2)

            with gcol_left:
                st.markdown("**Ground Truth + Missed**")
                st.iframe(_build_graph(data, diff, mode="left", **physics), height=640)

            with gcol_right:
                st.markdown("**Extracted (original + extra)**")
                st.iframe(_build_graph(data, diff, mode="right", **physics), height=640)
