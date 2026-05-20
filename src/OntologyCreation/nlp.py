import unicodedata

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_lemmatizer = WordNetLemmatizer()


def lemma_key(phrase: str) -> str:
    ascii_phrase = unicodedata.normalize("NFKD", phrase).encode("ascii", "ignore").decode("ascii")
    return " ".join(_lemmatizer.lemmatize(w) for w in ascii_phrase.lower().split())


def term_variants(term: str) -> list[str]:
    """Return term + its lemma form + first WordNet synset lemma names."""
    variants = {term.lower()}
    key = lemma_key(term)
    variants.add(key)
    synsets = wordnet.synsets(key.replace(" ", "_"))
    if synsets:
        for lemma in synsets[0].lemmas():  # pyright: ignore[reportOptionalMemberAccess]
            variants.add(lemma.name().replace("_", " ").lower())
    return list(variants)


def build_lemma_map(extracted: set[str], real: set[str]) -> dict[str, str]:
    """Map extracted names to real DBpedia names. Priority: exact < lemma < synset."""
    real_by_lemma: dict[str, str] = {}
    for r in real:
        key = lemma_key(r)
        real_by_lemma[key] = r
        real_by_lemma[key.replace(" ", "")] = r

    result = {}
    for e in extracted:
        if e in real:
            continue

        key = lemma_key(e)
        match = real_by_lemma.get(key) or real_by_lemma.get(key.replace(" ", ""))
        if match:
            result[e] = match
            continue

        synsets = wordnet.synsets(key.replace(" ", "_"))
        if synsets:
            for lm in synsets[0].lemmas():  # pyright: ignore[reportOptionalMemberAccess]
                syn_key = lemma_key(lm.name().replace("_", " "))
                match = real_by_lemma.get(syn_key) or real_by_lemma.get(syn_key.replace(" ", ""))
                if match:
                    result[e] = match
                    break

    return result


def is_mentioned(term_a: str, summary_b: str, term_b: str, summary_a: str) -> bool:
    """True if any variant of term_a appears in summary_b, or vice versa."""
    summary_a_l = summary_a.lower()
    summary_b_l = summary_b.lower()
    return any(v in summary_b_l for v in term_variants(term_a)) or any(
        v in summary_a_l for v in term_variants(term_b)
    )
