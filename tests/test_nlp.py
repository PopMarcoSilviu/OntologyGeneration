from OntologyCreation.nlp import build_lemma_map, is_mentioned, lemma_key


def test_lemma_key_lowercases():
    assert lemma_key("Dog") == "dog"


def test_lemma_key_lemmatizes_plural():
    assert lemma_key("dogs") == "dog"


def test_lemma_key_strips_accents():
    assert lemma_key("café") == "cafe"


def test_build_lemma_map_exact_already_in_real_skipped():
    assert build_lemma_map({"dog"}, {"dog"}) == {}


def test_build_lemma_map_plural_maps_to_canonical():
    result = build_lemma_map({"dogs"}, {"Dog"})
    assert result == {"dogs": "Dog"}


def test_build_lemma_map_no_match_returns_empty():
    assert build_lemma_map({"xyz123abc"}, {"Dog"}) == {}


def test_is_mentioned_term_a_found_in_summary_b():
    assert is_mentioned("dog", "some text", "mammal", "dogs are mammals")


def test_is_mentioned_term_b_found_in_summary_a():
    assert is_mentioned("mammal", "dogs are mammals", "dog", "some text")


def test_is_mentioned_neither():
    assert not is_mentioned("cat", "fish swim", "bird", "fish swim")
