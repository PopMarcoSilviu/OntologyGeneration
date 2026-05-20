from __future__ import annotations

from functools import reduce
from typing import overload

from pydantic import BaseModel


class ClassesOnly(BaseModel):
    classes: list[str] = []

    @overload
    def __or__(self, other: ClassesOnly) -> ClassesOnly: ...
    @overload
    def __or__(self, other: HierarchyOnly | Ontology) -> Ontology: ...
    def __or__(self, other):
        if isinstance(other, ClassesOnly):
            return ClassesOnly(classes=list(set(self.classes + other.classes)))
        if isinstance(other, HierarchyOnly):
            return Ontology(classes=self.classes, subclass_of=other.subclass_of)
        # Ontology
        return Ontology(
            classes=list(set(self.classes + other.classes)),
            subclass_of=other.subclass_of,
        )


class HierarchyOnly(BaseModel):
    subclass_of: dict[str, list[str]] = {}

    @overload
    def __or__(self, other: HierarchyOnly) -> HierarchyOnly: ...
    @overload
    def __or__(self, other: ClassesOnly | Ontology) -> Ontology: ...
    def __or__(self, other):
        if isinstance(other, HierarchyOnly):
            merged = {k: list(v) for k, v in self.subclass_of.items()}
            for k, v in other.subclass_of.items():
                merged.setdefault(k, []).extend(v)
            return HierarchyOnly(subclass_of=merged)
        if isinstance(other, ClassesOnly):
            return Ontology(classes=other.classes, subclass_of=self.subclass_of)
        # Ontology
        merged = {k: list(v) for k, v in self.subclass_of.items()}
        for k, v in other.subclass_of.items():
            merged.setdefault(k, []).extend(v)
        return Ontology(classes=other.classes, subclass_of=merged)


class Ontology(BaseModel):
    classes: list[str] = []
    subclass_of: dict[str, list[str]] = {}  # child -> [parents]

    def __or__(self, other: Ontology | ClassesOnly | HierarchyOnly) -> Ontology:
        if isinstance(other, HierarchyOnly):
            merged = {k: list(v) for k, v in self.subclass_of.items()}
            for k, v in other.subclass_of.items():
                merged.setdefault(k, []).extend(v)
            return Ontology(classes=self.classes, subclass_of=merged)
        if isinstance(other, ClassesOnly):
            return Ontology(
                classes=list(set(self.classes + other.classes)),
                subclass_of=self.subclass_of,
            )
        # Ontology | Ontology
        merged_classes = list(set(self.classes + other.classes))
        merged_hier = {k: list(v) for k, v in self.subclass_of.items()}
        for k, v in other.subclass_of.items():
            merged_hier.setdefault(k, []).extend(v)
        return Ontology(classes=merged_classes, subclass_of=merged_hier)


def merge_classes(results: list[ClassesOnly]) -> ClassesOnly:
    return reduce(lambda a, b: a | b, results, ClassesOnly())


def merge_hierarchy(results: list[HierarchyOnly]) -> HierarchyOnly:
    return reduce(lambda a, b: a | b, results, HierarchyOnly())


def merge(results: list[Ontology]) -> Ontology:
    return reduce(lambda a, b: a | b, results, Ontology())
