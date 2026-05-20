from OntologyCreation.models import ClassesOnly, HierarchyOnly, Ontology


def test_merge_classes():
    a = ClassesOnly(classes=["dog", "cat"])
    b = ClassesOnly(classes=["cat", "fish"])
    result = a | b
    assert isinstance(result, ClassesOnly)
    assert set(result.classes) == {"dog", "cat", "fish"}


def test_merge_hierarchy():
    a = HierarchyOnly(subclass_of={"dog": ["mammal"]})
    b = HierarchyOnly(subclass_of={"cat": ["mammal"]})
    result = a | b
    assert isinstance(result, HierarchyOnly)
    assert result.subclass_of == {"dog": ["mammal"], "cat": ["mammal"]}


def test_classes_or_hierarchy_gives_ontology():
    c = ClassesOnly(classes=["dog"])
    h = HierarchyOnly(subclass_of={"dog": ["mammal"]})
    result = c | h
    assert isinstance(result, Ontology)
    assert result.classes == ["dog"]
    assert result.subclass_of == {"dog": ["mammal"]}
