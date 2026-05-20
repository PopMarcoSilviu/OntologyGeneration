# Metrics Reference

> All comparisons are **case-insensitive**.

---

## Class Metrics

### Precision

$$\text{class\_precision} = \frac{|\text{extracted} \cap \text{original}|}{|\text{extracted}|}$$

Of all classes the model predicted, what fraction exist in the original ontology.

### Recall

$$\text{class\_recall} = \frac{|\text{extracted} \cap \text{original}|}{|\text{original}|}$$

Of all classes in the original ontology, what fraction the model found.

### Artifact breakdown (`classes`)

| Key | Description |
|-----|-------------|
| `in_original` | Predicted classes that also exist in the original ontology |
| `extra` | Predicted classes **not** in the original ontology *(may still be valid concepts)* |
| `missing` | Classes in the original ontology the model did **not** predict |

---

## Hierarchy Metrics

Each relationship is a `(child, parent)` pair — e.g. `dog → mammal`.

### Precision

$$\text{hierarchy\_precision} = \frac{|\text{extracted\_pairs} \cap \text{original\_pairs}|}{|\text{extracted\_pairs}|}$$

Of all pairs the model predicted, what fraction match the original ontology.

### Recall

$$\text{hierarchy\_recall} = \frac{|\text{extracted\_pairs} \cap \text{original\_pairs}|}{|\text{original\_pairs}|}$$

Of all pairs in the original ontology, what fraction the model found.

### Recall (mentionable)

$$\text{hierarchy\_recall\_mentionable} = \frac{|\text{extracted\_pairs} \cap \text{mentionable}|}{|\text{mentionable}|}$$

Recall restricted to **mentionable pairs** — see [[#Mentionable Pairs]].

### Artifact breakdown (`hierarchy`)

| Key | Description |
|-----|-------------|
| `in_original` | Predicted pairs that match the original ontology |
| `extra` | Predicted pairs **not** in the original ontology |
| `missing` | Original pairs the model did **not** predict |
| `mentionable_found` | Mentionable pairs correctly found |
| `mentionable_missing` | Mentionable pairs the model missed |

---

## Mentionable Pairs

A ground truth pair $(child, parent)$ is **mentionable** if:

- the *child's* Wikipedia summary mentions the parent name, **or**
- the *parent's* Wikipedia summary mentions the child name

*(case-insensitive string match)*

This is pre-computed at data-fetch time and stored in the hierarchy JSON:

```json
"Dog": { "parent": "Mammal", "mentioned": true }
```

### Why it matters

Some ground truth relationships cannot be inferred from Wikipedia text alone — the article simply never names the parent class. `hierarchy_recall_mentionable` measures only the pairs where **textual evidence existed**, giving a fairer picture of model performance.

> **Example:** if `mammal → animal` is in the ground truth but neither Wikipedia article ever uses the word *animal* or *mammal* in reference to each other, penalising the model for missing it is unfair.

---

## MLflow Logged Values

### Metrics

| Key | Formula |
|-----|---------|
| `class_precision` | see above |
| `class_recall` | see above |
| `hierarchy_precision` | see above |
| `hierarchy_recall` | see above |
| `hierarchy_recall_mentionable` | see above |
| `in_original_classes` | $\lvert\text{classes.in\_original}\rvert$ |
| `extra_classes` | $\lvert\text{classes.extra}\rvert$ |
| `missing_classes` | $\lvert\text{classes.missing}\rvert$ |

### Artifacts

| Path | Contents |
|------|----------|
| `diff/results.json` | Full breakdown dict (classes + hierarchy keys above) |
| `diff/summary.txt` | Human-readable text view of the same data |
