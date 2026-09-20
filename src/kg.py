"""
kg.py
-----
Chargement et manipulation du graphe de connaissances (Knowledge Graph)
utilisé par l'assistant GraphRAG.

Le graphe est stocké sous forme de fichier JSON (nodes/edges) et chargé
dans un networkx.MultiDiGraph pour permettre plusieurs relations entre
deux mêmes nœuds (ex: un ErrorCode peut avoir plusieurs causes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass
class Node:
    id: str
    type: str
    label: str
    text: str


class KnowledgeGraph:
    """Encapsule un networkx.MultiDiGraph représentant le domaine support technique."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeGraph":
        kg = cls()
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))

        for n in data["nodes"]:
            kg.graph.add_node(
                n["id"], type=n["type"], label=n["label"], text=n["text"]
            )

        for e in data["edges"]:
            kg.graph.add_edge(
                e["source"],
                e["target"],
                relation=e["relation"],
                confidence=float(e.get("confidence", 0.5)),
            )
        return kg

    def node(self, node_id: str) -> Node:
        d = self.graph.nodes[node_id]
        return Node(id=node_id, type=d["type"], label=d["label"], text=d["text"])

    def find_nodes_by_text(
        self, query: str, node_types: list[str] | None = None, search_full_text: bool = False
    ) -> list[str]:
        """Recherche naïve par mot-clé pour l'ancrage d'entités (entity linking).

        Par défaut, ne recherche que dans le *label* et l'*id* du nœud (pas
        dans le champ `text` libre), afin d'éviter les faux positifs quand
        un nœud mentionne un autre équipement dans sa description.
        Dans une version production, on remplacerait ceci par une recherche
        vectorielle (embeddings) ou un NER dédié.
        """
        # Mots génériques à ignorer pour éviter les faux positifs (ex: "équipement"
        # apparaît dans le label de TOUS les équipements et ne les distingue pas).
        stopwords = {
            "the", "and", "pour", "avec", "mon", "ma", "mes", "les", "des",
            "une", "un", "du", "de", "la", "le", "apres", "après", "mise",
            "jour", "affiche", "pourquoi", "equipement", "équipement", "equipment",
        }
        query_tokens = {
            tok for tok in _tokenize(query.lower()) if len(tok) > 2 and tok not in stopwords
        }
        matches = []
        for node_id, d in self.graph.nodes(data=True):
            if node_types and d["type"] not in node_types:
                continue
            haystack = f'{d["label"]} {node_id}'.lower()
            if search_full_text:
                haystack += f' {d["text"]}'.lower()
            haystack_tokens = {
                tok for tok in _tokenize(haystack) if tok not in stopwords
            }
            if query_tokens & haystack_tokens:
                matches.append(node_id)
        return matches

    def neighbors_with_edges(self, node_id: str):
        """Retourne les arêtes sortantes (target, relation, confidence)."""
        for _, target, data in self.graph.out_edges(node_id, data=True):
            yield target, data["relation"], data["confidence"]


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-zA-Z0-9àâäéèêëïîôöùûüç]+", text)
