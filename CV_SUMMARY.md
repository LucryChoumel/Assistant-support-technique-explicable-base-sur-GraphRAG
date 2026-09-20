# Résumé pour CV / Résumé pour CV

## Version française

**Assistant support technique explicable basé sur GraphRAG (Python, NetworkX, scikit-learn)**
Conception d'un pipeline de retrieval-augmented generation reposant sur un graphe de
connaissances plutôt qu'une recherche vectorielle classique, avec traçabilité complète
de chaque réponse jusqu'à un chemin explicite du graphe. Mise en place d'un protocole
d'évaluation quantitative comparant l'approche à une baseline RAG vectorielle (TF-IDF)
sur un jeu de questions annoté : +30 à +40 points d'exactitude et taux d'hallucination
divisé par deux. Bug de raisonnement multi-sauts identifié et corrigé grâce à cette
évaluation, illustrant une démarche itérative de recherche rigoureuse (tests unitaires,
analyse d'erreurs, compromis précision/rappel documenté).

## English version

**Explainable GraphRAG Assistant for Enterprise Technical Support (Python, NetworkX, scikit-learn)**
Built a graph-based retrieval-augmented generation pipeline for technical troubleshooting,
with every answer traceable to an explicit multi-hop path in a knowledge graph. Designed a
quantitative evaluation comparing the approach against a classic vector-RAG baseline
(TF-IDF) on an annotated question set, showing 30–40 point accuracy gains and roughly
half the hallucination rate. Diagnosed and fixed a multi-hop reasoning depth bug surfaced
by the evaluation itself, demonstrating a rigorous, iterative research methodology (unit
tests, error analysis, documented precision/recall trade-offs).
