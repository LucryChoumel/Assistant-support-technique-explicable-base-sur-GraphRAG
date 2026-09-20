"""
retriever.py
------------
Étape "Retrieval" du pipeline GraphRAG :
1. Extraction d'entités simples depuis la question (équipement, version, code erreur).
2. Ancrage (entity linking) de ces entités sur des nœuds du graphe.
3. Parcours du graphe (BFS borné) à partir des nœuds ancrés pour collecter
   les chemins d'évidence (evidence chains) jusqu'aux causes / procédures / documents.
4. Scoring des chemins par produit des confidences des arêtes traversées,
   pour classer les explications les plus fiables en premier.

C'est ce mécanisme de parcours explicite (par opposition à une simple
recherche vectorielle sur des chunks de texte) qui rend le système
"explicable" : chaque réponse est justifiée par un chemin traçable dans
le graphe de connaissances.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kg import KnowledgeGraph


@dataclass
class EvidenceStep:
    source_id: str
    relation: str
    target_id: str
    confidence: float


@dataclass
class EvidenceChain:
    steps: list[EvidenceStep] = field(default_factory=list)

    @property
    def score(self) -> float:
        score = 1.0
        for s in self.steps:
            score *= s.confidence
        return score

    @property
    def end_node(self) -> str:
        return self.steps[-1].target_id if self.steps else ""


@dataclass
class RetrievalResult:
    anchored_entities: list[str]
    chains: list[EvidenceChain]


ERROR_CODE_RE = re.compile(r"\berror\s*[:#]?\s*(\d{2,5})\b", re.IGNORECASE)
VERSION_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")


def extract_entities(query: str) -> dict[str, str | None]:
    """Extraction légère d'entités par expressions régulières.

    Dans un système en production, on utiliserait un modèle NER /
    un LLM pour une extraction plus robuste (synonymes, fautes de frappe...).
    """
    error_match = ERROR_CODE_RE.search(query)
    version_match = VERSION_RE.search(query)
    return {
        "error_code": error_match.group(1) if error_match else None,
        "version": version_match.group(1) if version_match else None,
        "raw_query": query,
    }


class GraphRetriever:
    def __init__(self, kg: KnowledgeGraph, max_hops: int = 4, top_k: int = 5):
        self.kg = kg
        self.max_hops = max_hops
        self.top_k = top_k

    def anchor_entities(self, query: str) -> list[str]:
        entities = extract_entities(query)
        anchored: list[str] = []

        # Ancrage précis par identifiant reconstruit (ex: "542" -> "error:542").
        # On préfère un lookup exact plutôt qu'une recherche floue quand le
        # code erreur / la version sont explicitement mentionnés, pour éviter
        # les faux positifs entre codes d'erreur ou versions différents.
        if entities["error_code"]:
            candidate = f'error:{entities["error_code"]}'
            if self.kg.graph.has_node(candidate):
                anchored.append(candidate)
        if entities["version"]:
            candidate = f'fw:{entities["version"]}'
            if self.kg.graph.has_node(candidate):
                anchored.append(candidate)

        # Fallback : recherche par mots-clés (nom/modèle d'équipement, etc.)
        anchored += self.kg.find_nodes_by_text(query, node_types=["Equipment"])

        # dédoublonnage en conservant l'ordre
        seen = set()
        result = []
        for a in anchored:
            if a not in seen:
                seen.add(a)
                result.append(a)
        return result

    def expand(self, start_nodes: list[str]) -> list[EvidenceChain]:
        """BFS borné collectant tous les chemins depuis les nœuds ancrés
        jusqu'à des nœuds "terminaux" utiles (Cause, FixProcedure, Document)."""
        terminal_types = {"Cause", "FixProcedure", "Document"}
        chains: list[EvidenceChain] = []

        for start in start_nodes:
            stack: list[EvidenceChain] = [EvidenceChain(steps=[])]
            frontier = [(start, EvidenceChain(steps=[]))]
            visited_paths = 0

            def dfs(node_id: str, chain: EvidenceChain, depth: int):
                nonlocal visited_paths
                if visited_paths > 200:
                    return
                node_type = self.kg.graph.nodes[node_id]["type"]
                if chain.steps and node_type in terminal_types:
                    chains.append(chain)
                if depth >= self.max_hops:
                    return
                for target, relation, confidence in self.kg.neighbors_with_edges(node_id):
                    visited_paths += 1
                    new_chain = EvidenceChain(steps=chain.steps + [
                        EvidenceStep(node_id, relation, target, confidence)
                    ])
                    dfs(target, new_chain, depth + 1)

            dfs(start, EvidenceChain(steps=[]), 0)

        # Classement par score de confiance décroissant, dédoublonnage par nœud final+relation
        chains.sort(key=lambda c: c.score, reverse=True)
        deduped: list[EvidenceChain] = []
        seen_signatures = set()
        seen_endpoints = set()
        for c in chains:
            sig = tuple((s.source_id, s.relation, s.target_id) for s in c.steps)
            # on ne garde que la meilleure chaîne (score le plus élevé, car déjà
            # triées) menant à un nœud terminal donné, pour éviter les doublons
            # de chemins redondants vers la même Cause/Fix/Document.
            if sig in seen_signatures or c.end_node in seen_endpoints:
                continue
            seen_signatures.add(sig)
            seen_endpoints.add(c.end_node)
            deduped.append(c)

        # Sélection diversifiée : on garantit qu'au moins une chaîne de chaque
        # type terminal (Cause, FixProcedure, Document) apparaisse dans le
        # top_k si elle existe, plutôt que de laisser un seul type saturer
        # le classement.
        by_type: dict[str, list[EvidenceChain]] = {}
        for c in deduped:
            node_type = self.kg.graph.nodes[c.end_node]["type"]
            by_type.setdefault(node_type, []).append(c)

        selected: list[EvidenceChain] = []
        for node_type in ("Cause", "FixProcedure", "Document"):
            if by_type.get(node_type):
                selected.append(by_type[node_type][0])

        for c in deduped:
            if len(selected) >= self.top_k:
                break
            if c not in selected:
                selected.append(c)

        return selected[: self.top_k]

    def retrieve(self, query: str) -> RetrievalResult:
        anchored = self.anchor_entities(query)
        chains = self.expand(anchored)
        return RetrievalResult(anchored_entities=anchored, chains=chains)
