"""
baseline_tfidf.py
------------------
Baseline "RAG vectoriel classique" utilisée pour comparer quantitativement
l'approche GraphRAG à une approche standard : chaque nœud du graphe est traité
comme un simple chunk de texte indépendant (on ignore les relations du graphe),
et la récupération se fait par similarité cosinus sur une représentation
TF-IDF de la question et des chunks — c'est le comportement typique d'un RAG
"à plat" sur des documents non structurés (sans multi-hop reasoning).

Cette baseline reçoit volontairement un avantage (elle a accès au type de
chaque chunk : Cause / FixProcedure) pour rester une comparaison honnête et
non caricaturale : la différence mesurée provient donc bien de l'absence de
raisonnement multi-sauts sur un graphe, pas d'un désavantage artificiel.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kg import KnowledgeGraph

SIMILARITY_THRESHOLD = 0.12  # en dessous de ce score, la baseline s'abstient


@dataclass
class BaselinePrediction:
    predicted_cause: str | None
    predicted_fix: str | None
    best_cause_score: float
    best_fix_score: float


class TfidfBaseline:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.node_ids = list(kg.graph.nodes())
        corpus = [
            f'{kg.graph.nodes[n]["label"]} {kg.graph.nodes[n]["text"]}'
            for n in self.node_ids
        ]
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def _best_match(self, query: str, node_type: str) -> tuple[str | None, float]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix)[0]

        best_id, best_score = None, 0.0
        for node_id, score in zip(self.node_ids, sims):
            if self.kg.graph.nodes[node_id]["type"] != node_type:
                continue
            if score > best_score:
                best_id, best_score = node_id, score

        if best_score < SIMILARITY_THRESHOLD:
            return None, best_score
        return best_id, best_score

    def predict(self, question: str) -> BaselinePrediction:
        cause_id, cause_score = self._best_match(question, "Cause")
        fix_id, fix_score = self._best_match(question, "FixProcedure")
        return BaselinePrediction(
            predicted_cause=cause_id,
            predicted_fix=fix_id,
            best_cause_score=cause_score,
            best_fix_score=fix_score,
        )
