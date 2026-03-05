from hallucination_scorer.sentence_splitter import split_into_sentences


def test_split_into_sentences_basic():
    text = "The Eiffel Tower is in Paris. It was completed in 1889."
    sentences = split_into_sentences(text)
    assert sentences == [
        "The Eiffel Tower is in Paris.",
        "It was completed in 1889.",
    ]


def test_split_into_sentences_handles_abbreviations():
    text = "Dr. Smith went to Paris. He visited the Eiffel Tower."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].endswith("Paris.")
    assert sentences[1].startswith("He visited")

