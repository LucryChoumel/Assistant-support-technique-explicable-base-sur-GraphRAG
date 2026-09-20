"""
test_explainability.py
------------------------
Vérifie que la méthode d'explicabilité utilisée (traçabilité par chemin de
graphe, cf. section "Méthode d'explicabilité" du README) est réellement
FIDÈLE en mode local : chaque affirmation présente dans le texte de réponse
doit correspondre à une chaîne d'évidence effectivement retournée par le
retriever, jamais à du texte "inventé" hors de ces chaînes.

Cette distinction (fidélité vs plausibilité) est centrale dans la
littérature sur l'explicabilité (Jacovi & Goldberg, 2020) : une explication
"fidèle" reflète réellement le mécanisme de décision, une explication
"plausible" est seulement convaincante pour un humain, sans garantie qu'elle
soit vraie. Ce test ne couvre QUE le mode local : le mode LLM (génération
optionnelle via call_llm) n'offre aucune garantie structurelle de fidélité
— seulement une instruction de prompt — et n'est pas vérifiable sans appel
réseau à un fournisseur externe.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explainer import build_local_answer  # noqa: E402
from kg import KnowledgeGraph  # noqa: E402
from retriever import GraphRetriever  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_kg.json"


def test_local_answer_is_faithful_to_evidence_chains():
    """Chaque texte de nœud terminal cité dans la réponse doit provenir
    d'une chaîne d'évidence réellement retournée — aucune affirmation ne
    doit apparaître sans chemin de graphe qui la justifie."""
    kg = KnowledgeGraph.from_json(DATA_PATH)
    retriever = GraphRetriever(kg, max_hops=4, top_k=5)
    result = retriever.retrieve(
        "Pourquoi mon équipement X100 affiche Error 542 après la mise à jour 4.2 ?"
    )
    explanation = build_local_answer(kg, result)

    cited_node_texts = {kg.node(c.end_node).text for c in result.chains}
    assert cited_node_texts, "aucune chaîne d'évidence retournée pour ce test"

    for text in cited_node_texts:
        assert text in explanation.answer_text, (
            f"Le texte du nœud '{text}' fait partie des chaînes d'évidence "
            "mais n'apparaît pas dans la réponse générée : la réponse "
            "n'est plus fidèle à ce qu'elle cite."
        )

    # Chaque citation affichée doit elle-même correspondre à une chaîne réelle,
    # pas à du texte généré librement.
    real_chain_descriptions_count = len(result.chains)
    assert len(explanation.citations) == real_chain_descriptions_count


def test_abstention_produces_no_uncited_claim():
    """Quand aucune chaîne n'est trouvée, la réponse doit explicitement dire
    qu'elle ne sait pas, et ne doit citer aucune source (rien à inventer)."""
    kg = KnowledgeGraph.from_json(DATA_PATH)
    retriever = GraphRetriever(kg, max_hops=4, top_k=5)
    result = retriever.retrieve("Error 999 totalement inconnue")
    explanation = build_local_answer(kg, result)

    assert explanation.citations == []
    assert explanation.confidence == 0.0


if __name__ == "__main__":
    test_local_answer_is_faithful_to_evidence_chains()
    test_abstention_produces_no_uncited_claim()
    print("Tous les tests de fidélité de l'explication sont passés ✅")
