import pytest

from OntologyCreation.models import Ontology
from OntologyCreation.test import evaluate


def test_perfect_extraction():
    gt = Ontology(classes=["dog", "cat"], subclass_of={"dog": ["mammal"], "cat": ["mammal"]})
    ext = Ontology(classes=["dog", "cat"], subclass_of={"dog": ["mammal"], "cat": ["mammal"]})
    metrics, _ = evaluate(ext, gt, {("dog", "mammal"), ("cat", "mammal")})
    assert metrics["class_precision"] == 1.0
    assert metrics["class_recall"] == 1.0
    assert metrics["hierarchy_precision"] == 1.0
    assert metrics["hierarchy_recall"] == 1.0
    assert metrics["hierarchy_recall_mentionable"] == 1.0


def test_empty_extraction_gives_zero_recall():
    gt = Ontology(classes=["dog", "cat"], subclass_of={})
    metrics, _ = evaluate(Ontology(classes=[], subclass_of={}), gt, set())
    assert metrics["class_precision"] == 0.0
    assert metrics["class_recall"] == 0.0


def test_partial_extraction_precision_and_recall():
    gt = Ontology(classes=["dog", "cat", "fish"], subclass_of={})
    ext = Ontology(classes=["dog", "cat"], subclass_of={})
    metrics, _ = evaluate(ext, gt, set())
    assert metrics["class_precision"] == 1.0
    assert metrics["class_recall"] == pytest.approx(2 / 3)


def test_extra_classes_lower_precision():
    gt = Ontology(classes=["dog"], subclass_of={})
    ext = Ontology(classes=["dog", "alien"], subclass_of={})
    metrics, _ = evaluate(ext, gt, set())
    assert metrics["class_precision"] == 0.5
    assert metrics["class_recall"] == 1.0


def test_details_has_required_keys():
    gt = Ontology(classes=["dog"], subclass_of={"dog": ["mammal"]})
    ext = Ontology(classes=["dog"], subclass_of={"dog": ["mammal"]})
    _, details = evaluate(ext, gt, set())
    assert set(details["classes"]) == {"in_original", "extra", "missing"}
    assert {"in_original", "extra", "missing"}.issubset(details["hierarchy"])


def test_details_in_original_and_extra():
    gt = Ontology(classes=["dog"], subclass_of={})
    ext = Ontology(classes=["dog", "alien"], subclass_of={})
    _, details = evaluate(ext, gt, set())
    assert "dog" in details["classes"]["in_original"]
    assert "alien" in details["classes"]["extra"]
    assert details["classes"]["missing"] == []


def test_fuzzy_match_plural_counts_as_correct():
    gt = Ontology(classes=["Dog"], subclass_of={})
    ext = Ontology(classes=["dogs"], subclass_of={})
    metrics, _ = evaluate(ext, gt, set(), strict=False)
    assert metrics["class_recall"] == 1.0


def test_strict_mode_disables_fuzzy_match():
    gt = Ontology(classes=["Dog"], subclass_of={})
    ext = Ontology(classes=["dogs"], subclass_of={})
    metrics, _ = evaluate(ext, gt, set(), strict=True)
    assert metrics["class_recall"] == 0.0


def test_mentionable_recall_independent_of_full_hierarchy_recall():
    gt = Ontology(classes=["dog", "cat"], subclass_of={"dog": ["mammal"], "cat": ["mammal"]})
    ext = Ontology(classes=["dog"], subclass_of={"dog": ["mammal"]})
    mentionable = {("dog", "mammal")}
    metrics, _ = evaluate(ext, gt, mentionable)
    assert metrics["hierarchy_recall_mentionable"] == 1.0
    assert metrics["hierarchy_recall"] == pytest.approx(0.5)
