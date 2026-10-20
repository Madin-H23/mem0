import pytest


@pytest.fixture(autouse=True)
def _ensure_spacy():
    """Skip tests if spaCy model is not available."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("spaCy en_core_web_sm model not available")


class TestLemmatizeForBm25:
    def test_basic_lemmatization(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("The cats are running quickly")
        assert "cat" in result
        assert "run" in result or "running" in result
        # Stop words and punctuation should be removed
        assert "the" not in result.split()

    def test_verb_forms_normalized(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("she attended multiple meetings yesterday")
        assert "attend" in result or "attended" in result
        assert "meeting" in result  # -ing form preserved alongside lemma
        # "multiple" is kept (not a spaCy stop word)

    def test_ing_preservation(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("attending the morning meeting")
        tokens = result.split()
        # Should have both the lemma and the -ing form
        assert "attending" in tokens or "attend" in tokens

    def test_empty_string(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("")
        assert result == ""

    def test_punctuation_removed(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_lowercased(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("PYTHON Programming LANGUAGE")
        for token in result.split():
            assert token == token.lower()

    def test_stop_words_removed(self):
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("this is a very simple test of the system")
        tokens = result.split()
        for stop in ["this", "is", "a", "very", "of", "the"]:
            assert stop not in tokens

    def test_alphanumeric_hyphenated_tokens_kept(self):
        """Model names and version numbers (GPT-4, COVID-19) are single
        hyphenated tokens and must stay in the BM25 index."""
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("I use GPT-4 and COVID-19 data")
        tokens = result.split()
        assert "gpt-4" in tokens
        assert "covid-19" in tokens

    def test_plain_word_still_indexed(self):
        """Ordinary words must keep round-tripping through the lemma index."""
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("I remember multiple memories of the system")
        tokens = result.split()
        assert "memory" in tokens
        assert "system" in tokens
        # A lone plain word survives untouched.
        assert lemmatize_for_bm25("meeting").split() == ["meeting"]

    def test_model_name_variants_kept(self):
        """Hyphenated model names in their real spellings stay single terms."""
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("compare mixtral-8x7b with gpt-4o and text-embedding-3")
        tokens = result.split()
        assert "mixtral-8x7b" in tokens
        assert "gpt-4o" in tokens
        # The hyphen is never split: the term stays whole, no bare fragments.
        assert "8x7b" not in tokens
        assert "4o" not in tokens

    def test_non_alphanumeric_lemmas_dropped(self):
        """The widened predicate still drops lemmas that are not alphanumeric
        even after removing hyphens (abbreviations, handles, decimals)."""
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("see e.g. u.s.a @user 3.14 details")
        tokens = result.split()
        for junk in ["e.g.", "u.s.a", "@us", "3.14"]:
            assert junk not in tokens
        # The real words around them are unaffected.
        assert "detail" in tokens or "details" in tokens

    def test_hyphenated_phrase_words_kept(self):
        """spaCy splits hyphenated phrases (state-of-the-art) into words: the
        words stay indexed and the bare hyphens never reach the index."""
        from mem0.utils.lemmatization import lemmatize_for_bm25

        result = lemmatize_for_bm25("state-of-the-art tooling here")
        tokens = result.split()
        assert "state" in tokens and "art" in tokens
        assert "-" not in tokens
