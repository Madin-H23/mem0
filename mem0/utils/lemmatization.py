"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def lemmatize_for_bm25(text: str) -> str:
    """Lemmatize text for BM25 matching.

    Returns space-joined lemmas for full-text search. Falls back to
    the original text if spaCy is unavailable.
    """
    from mem0.utils.spacy_models import get_nlp_lemma

    nlp = get_nlp_lemma()
    if nlp is None:
        return text

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        # Keep alphanumeric lemmas and hyphenated ones: spaCy hands us GPT-4,
        # COVID-19 and mixtral-8x7b as single tokens whose lemma fails
        # isalnum() only because of the hyphen, and dropping them removes the
        # most searched terms (model names, version numbers) from the BM25
        # index. Never split on the hyphen: gpt-4 as one term matches a gpt-4
        # query exactly, while gpt + 4 collides with every bare number in the
        # corpus. Lemmas that stay non-alphanumeric without a hyphen (e.g.,
        # u.s.a, @user) are still dropped, and spaCy's lemma_ can re-case
        # known words, so lowercase before indexing to keep both sides equal.
        if lemma.isalnum() or lemma.replace("-", "").isalnum():
            tokens.append(lemma.lower())

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if (
            token.text.endswith("ing")
            and token.text != lemma
            and (token.text.isalnum() or token.text.replace("-", "").isalnum())
        ):
            tokens.append(token.text)

    return " ".join(tokens)
