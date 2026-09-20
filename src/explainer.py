"""
explainer.py
------------
Étape "Generation" du pipeline GraphRAG.

Transforme les chaînes d'évidence (EvidenceChain) récupérées dans le graphe
en une réponse en langage naturel, accompagnée de sa justification explicite
(le chemin du graphe qui a produit chaque affirmation).

Deux modes :
- Mode local (par défaut, aucune dépendance réseau) : un gabarit déterministe
  assemble la réponse directement depuis les nœuds/arêtes du graphe. Zéro
  hallucination possible puisqu'aucun texte n'est généré librement.
- Mode LLM (optionnel, activé en définissant la variable d'environnement
  `LLM_PROVIDER`) : le sous-graphe récupéré est injecté comme contexte dans
  un appel à un modèle de langage générique, avec une consigne stricte de ne
  pas sortir du contexte fourni ("closed-book grounding"), afin de reformuler
  la réponse en langage plus naturel tout en restant traçable. L'architecture
  est volontairement indépendante d'un fournisseur particulier : il suffit
  d'implémenter la fonction `call_llm()` pour le fournisseur de son choix
  (API OpenAI-compatible, Anthropic, Mistral, modèle local via Ollama, etc.).

Dans les deux cas, chaque affirmation de la réponse reste traçable jusqu'à
un nœud/arête précis du graphe : c'est le cœur de l'"explicabilité" du système.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from kg import KnowledgeGraph
from retriever import EvidenceChain, RetrievalResult


@dataclass
class Explanation:
    answer_text: str
    citations: list[str]  # descriptions lisibles des chemins de preuve
    confidence: float


def _describe_chain(kg: KnowledgeGraph, chain: EvidenceChain) -> str:
    parts = []
    for step in chain.steps:
        src = kg.node(step.source_id).label
        tgt = kg.node(step.target_id).label
        parts.append(f"{src} --[{step.relation}]--> {tgt}")
    return "  →  ".join(parts)


def build_local_answer(kg: KnowledgeGraph, result: RetrievalResult) -> Explanation:
    """Génère une réponse sans appel LLM, uniquement à partir du graphe."""
    if not result.chains:
        return Explanation(
            answer_text=(
                "Je n'ai pas trouvé d'information suffisante dans la base de "
                "connaissances pour répondre avec certitude. Merci de préciser "
                "le modèle d'équipement et le code d'erreur exact."
            ),
            citations=[],
            confidence=0.0,
        )

    causes = [c for c in result.chains if kg.node(c.end_node).type == "Cause"]
    fixes = [c for c in result.chains if kg.node(c.end_node).type == "FixProcedure"]
    docs = [c for c in result.chains if kg.node(c.end_node).type == "Document"]

    lines = []
    if causes:
        lines.append("**Causes probables :**")
        for c in causes:
            cause_node = kg.node(c.end_node)
            lines.append(f"- {cause_node.text} (confiance ≈ {c.score:.0%})")

    if fixes:
        lines.append("\n**Actions recommandées :**")
        for f in fixes:
            fix_node = kg.node(f.end_node)
            lines.append(f"- {fix_node.text}")

    if docs:
        lines.append("\n**Sources documentaires :**")
        for d in docs:
            doc_node = kg.node(d.end_node)
            lines.append(f"- {doc_node.label} : {doc_node.text}")

    answer_text = "\n".join(lines)
    citations = [_describe_chain(kg, c) for c in result.chains]
    top_confidence = max((c.score for c in causes), default=(
        max((c.score for c in result.chains), default=0.0)
    ))

    return Explanation(answer_text=answer_text, citations=citations, confidence=top_confidence)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Point d'intégration générique vers un modèle de langage externe.

    Volontairement découplé de tout fournisseur : le choix se fait via la
    variable d'environnement `LLM_PROVIDER` (ex: "openai", "anthropic",
    "local"...). Chaque adaptateur est isolé dans son propre bloc `if`, avec
    un import local, pour que l'ajout/retrait d'un fournisseur n'affecte pas
    le reste du pipeline. Aucun fournisseur n'est requis pour faire tourner
    le projet : sans configuration, `explain()` utilise le mode local.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "openai":
        import openai  # dépendance optionnelle

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    if provider == "anthropic":
        import anthropic  # dépendance optionnelle

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-6"),
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    raise NotImplementedError(
        f"Fournisseur LLM_PROVIDER={provider!r} non configuré. "
        "Implémentez un adaptateur dans call_llm() ou laissez la variable "
        "d'environnement vide pour utiliser le mode local."
    )


def build_llm_answer(kg: KnowledgeGraph, result: RetrievalResult, question: str) -> Explanation:
    """Version augmentée : reformule la réponse en langage plus naturel via
    un LLM générique (cf. `call_llm`), strictement bornée au contexte du
    graphe fourni pour limiter les hallucinations.
    """
    context_lines = [_describe_chain(kg, c) for c in result.chains]
    context_block = "\n".join(f"- {line}" for line in context_lines) or "(aucune évidence trouvée)"

    system_prompt = (
        "Tu es un assistant de support technique. Réponds UNIQUEMENT à partir "
        "des faits du graphe de connaissances fournis ci-dessous, sans ajouter "
        "d'information externe. Si le contexte est insuffisant, dis-le "
        "explicitement. Structure ta réponse en 'Causes probables' et "
        "'Actions recommandées', en citant les chemins du graphe utilisés."
    )
    user_prompt = (
        f"Question du client : {question}\n\n"
        f"Contexte extrait du graphe de connaissances :\n{context_block}"
    )

    text = call_llm(system_prompt, user_prompt)

    local = build_local_answer(kg, result)
    return Explanation(answer_text=text, citations=local.citations, confidence=local.confidence)


def explain(kg: KnowledgeGraph, result: RetrievalResult, question: str) -> Explanation:
    if os.environ.get("LLM_PROVIDER"):
        try:
            return build_llm_answer(kg, result, question)
        except Exception:
            # repli silencieux vers le mode local en cas d'erreur réseau/API/config
            pass
    return build_local_answer(kg, result)
