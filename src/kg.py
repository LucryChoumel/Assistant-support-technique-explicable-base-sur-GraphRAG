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

from nlp import extract_keywords


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
                n["id"],
                type=n["type"],
                label=n["label"],
                text=n["text"],
                # Mots-clés NLP pré-calculés une seule fois au chargement (et
                # non à chaque requête) : le coût du traitement NLP (lemmatisation
                # + POS tagging par spaCy) est ainsi payé une fois pour toutes,
                # pas pour chaque question posée au système.
                keywords_label=extract_keywords(f'{n["label"]} {n["id"]}'),
                keywords_full=extract_keywords(f'{n["label"]} {n["id"]} {n["text"]}'),
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
        """Ancrage d'entités (entity linking) par recoupement de mots-clés.

        Les mots-clés de la question et des nœuds sont extraits via
        `nlp.extract_keywords` : lemmatisation + filtrage grammatical par
        spaCy (mode NLP), avec repli automatique par regex si spaCy/le
        modèle ne sont pas disponibles (voir `src/nlp.py`). Les mots-clés des
        nœuds sont pré-calculés une seule fois à `from_json` (voir ci-dessus) ;
        seuls ceux de la question sont calculés à la volée, à chaque appel.

        Par défaut, ne recherche que dans le *label* et l'*id* du nœud (pas
        dans le champ `text` libre), afin d'éviter les faux positifs quand
        un nœud mentionne un autre équipement dans sa description.
        """
        query_keywords = extract_keywords(query)
        matches = []
        for node_id, d in self.graph.nodes(data=True):
            if node_types and d["type"] not in node_types:
                continue
            haystack_keywords = d["keywords_full"] if search_full_text else d["keywords_label"]
            if query_keywords & haystack_keywords:
                matches.append(node_id)
        return matches

    def neighbors_with_edges(self, node_id: str):
        """Retourne les arêtes sortantes (target, relation, confidence)."""
        for _, target, data in self.graph.out_edges(node_id, data=True):
            yield target, data["relation"], data["confidence"]
