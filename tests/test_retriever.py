import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kg import KnowledgeGraph  # noqa: E402
from retriever import GraphRetriever, extract_entities  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_kg.json"


def test_extract_entities_finds_error_and_version():
    entities = extract_entities(
        "Pourquoi mon équipement X100 affiche Error 542 après la mise à jour 4.2 ?"
    )
    assert entities["error_code"] == "542"
    assert entities["version"] == "4.2"


def test_retrieval_finds_cause_and_fix_for_error_542():
    kg = KnowledgeGraph.from_json(DATA_PATH)
    retriever = GraphRetriever(kg, max_hops=4, top_k=5)
    result = retriever.retrieve(
        "Pourquoi mon équipement X100 affiche Error 542 après la mise à jour 4.2 ?"
    )

    assert "error:542" in result.anchored_entities
    end_types = {kg.node(c.end_node).type for c in result.chains}
    assert "Cause" in end_types
    assert "FixProcedure" in end_types

    top_chain = result.chains[0]
    assert kg.node(top_chain.end_node).id in {
        "cause:cert_outdated",
        "fix:update_cert",
        "doc:kb_542",
    }


def test_unrelated_query_returns_no_confident_chain():
    kg = KnowledgeGraph.from_json(DATA_PATH)
    retriever = GraphRetriever(kg, max_hops=4, top_k=5)
    result = retriever.retrieve("Error 999 inconnue")
    assert result.chains == [] or all(c.score < 0.5 for c in result.chains)


if __name__ == "__main__":
    test_extract_entities_finds_error_and_version()
    test_retrieval_finds_cause_and_fix_for_error_542()
    test_unrelated_query_returns_no_confident_chain()
    print("Tous les tests sont passés ✅")
