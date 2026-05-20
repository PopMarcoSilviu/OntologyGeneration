def word_count(text: str) -> int:
    return len(text.split())


def chunk_by_words(items: list[tuple[str, str]], max_words: int) -> list[list[tuple[str, str]]]:
    """Split (name, summary) pairs into chunks where total words <= max_words."""
    chunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_words = 0
    for name, summary in items:
        w = word_count(summary)
        if current and current_words + w > max_words:
            chunks.append(current)
            current = []
            current_words = 0
        current.append((name, summary))
        current_words += w
    if current:
        chunks.append(current)
    return chunks


def format_diff(details: dict) -> str:
    lines = ["CLASSES"]
    for key in ("in_original", "extra", "missing"):
        items = details["classes"][key]
        lines.append(f"  {key:<12} ({len(items)}): {', '.join(items) or '-'}")

    lines.append("\nHIERARCHY")
    by_parent: dict[str, dict[str, list[str]]] = {}
    for bucket in ("in_original", "missing"):
        for pair in details["hierarchy"][bucket]:
            child, parent = pair.split(" -> ")
            by_parent.setdefault(parent, {"in_original": [], "missing": []})
            by_parent[parent][bucket].append(child)

    for parent in sorted(by_parent):
        lines.append(f"  {parent}")
        for bucket, marker in (("in_original", "✓"), ("missing", "✗")):
            children = sorted(by_parent[parent][bucket])
            if children:
                lines.append(f"    {marker} {bucket:<12}: {', '.join(children)}")

    lines.append("\nMENTIONABLE HIERARCHY")
    for bucket, marker in (("mentionable_found", "✓"), ("mentionable_missing", "✗")):
        items = details["hierarchy"][bucket]
        label = "found" if bucket == "mentionable_found" else "missing"
        lines.append(f"  {marker} {label:<8} ({len(items)}): {', '.join(items) or '-'}")

    extra = details["hierarchy"]["extra"]
    if extra:
        lines.append("\n  --- extra relationships (not in ground truth) ---")
        by_extra_parent: dict[str, list[str]] = {}
        for pair in extra:
            child, parent = pair.split(" -> ")
            by_extra_parent.setdefault(parent, []).append(child)
        for parent in sorted(by_extra_parent):
            lines.append(f"    {parent}: {', '.join(sorted(by_extra_parent[parent]))}")

    return "\n".join(lines)
