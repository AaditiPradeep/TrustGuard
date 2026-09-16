"""Dependency-free checks for dense-index corpus preparation."""

from data.schema import EvidencePassage
from retrieval.dense_index import chunk_passage


def main() -> None:
    passage = EvidencePassage(
        text="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        source_name="Test source",
        source_tier="reference",
        identifier="test-1",
    )
    chunks = chunk_passage(passage, chunk_size=24, chunk_overlap=8)
    assert len(chunks) > 1
    assert all(chunk.source_name == "Test source" for chunk in chunks)
    assert all(":chunk:" in chunk.identifier for chunk in chunks)
    assert "alpha" in chunks[0].text and "mu" in chunks[-1].text
    print(f"Created {len(chunks)} provenance-preserving chunks.")


if __name__ == "__main__":
    main()
