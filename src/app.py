"""
app.py
------
Point d'entrée en ligne de commande de l'assistant GraphRAG explicable.

Usage :
    python src/app.py "Pourquoi mon équipement X100 affiche Error 542 après la mise à jour 4.2 ?"

    ou en mode interactif :
    python src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from explainer import explain
from kg import KnowledgeGraph
from retriever import GraphRetriever

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_kg.json"


def answer_question(question: str, kg: KnowledgeGraph) -> None:
    retriever = GraphRetriever(kg, max_hops=4, top_k=5)
    result = retriever.retrieve(question)
    explanation = explain(kg, result, question)

    print("\n=== Réponse ===")
    print(explanation.answer_text)

    print("\n=== Traçabilité (explicabilité) ===")
    if explanation.citations:
        for i, citation in enumerate(explanation.citations, start=1):
            print(f"[{i}] {citation}")
    else:
        print("(aucun chemin d'évidence trouvé dans le graphe)")

    print(f"\nConfiance globale estimée : {explanation.confidence:.0%}")
    print(f"Entités ancrées dans le graphe : {result.anchored_entities}")


def main() -> None:
    kg = KnowledgeGraph.from_json(DATA_PATH)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer_question(question, kg)
        return

    print("Assistant support technique GraphRAG — tapez 'exit' pour quitter.\n")
    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        answer_question(question, kg)
        print()


if __name__ == "__main__":
    main()
