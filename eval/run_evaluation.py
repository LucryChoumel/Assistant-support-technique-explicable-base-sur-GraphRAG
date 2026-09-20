"""
run_evaluation.py
------------------
Évaluation quantitative comparant :
  (A) notre pipeline GraphRAG (parcours du graphe de connaissances)
  (B) une baseline RAG vectorielle classique (TF-IDF, chunks indépendants)

sur un même jeu de 10 questions annotées (eval/eval_dataset.json), incluant
des cas positifs (réponse attendue) et des cas négatifs (le système doit
s'abstenir plutôt que d'inventer une réponse).

Métriques calculées :
- Exactitude de la cause identifiée (Cause Accuracy)
- Exactitude du correctif identifié (Fix Accuracy)
- Taux de faux positifs sur les cas où le système doit s'abstenir
  (Hallucination Rate) — un système fiable doit répondre 0 sur ces cas
- Latence moyenne par requête (ms)

Usage :
    python eval/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from kg import KnowledgeGraph  # noqa: E402
from retriever import GraphRetriever  # noqa: E402
from baseline_tfidf import TfidfBaseline  # noqa: E402

DATA_PATH = ROOT / "data" / "sample_kg.json"
EVAL_PATH = ROOT / "eval" / "eval_dataset.json"


def graphrag_predict(kg: KnowledgeGraph, retriever: GraphRetriever, question: str):
    result = retriever.retrieve(question)
    predicted_cause = None
    predicted_fix = None
    for chain in result.chains:
        node_type = kg.graph.nodes[chain.end_node]["type"]
        if node_type == "Cause" and predicted_cause is None:
            predicted_cause = chain.end_node
        if node_type == "FixProcedure" and predicted_fix is None:
            predicted_fix = chain.end_node
    return predicted_cause, predicted_fix


def run():
    kg = KnowledgeGraph.from_json(DATA_PATH)
    retriever = GraphRetriever(kg, max_hops=4, top_k=5)
    baseline = TfidfBaseline(kg)
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["cases"]

    rows = []
    graphrag_latencies = []
    baseline_latencies = []

    for case in cases:
        q = case["question"]

        t0 = time.perf_counter()
        gr_cause, gr_fix = graphrag_predict(kg, retriever, q)
        graphrag_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        bl_pred = baseline.predict(q)
        baseline_latencies.append((time.perf_counter() - t0) * 1000)

        rows.append({
            "id": case["id"],
            "question": q,
            "expected_cause": case["expected_cause"],
            "expected_fix": case["expected_fix"],
            "should_abstain": case["should_abstain"],
            "graphrag_cause": gr_cause,
            "graphrag_fix": gr_fix,
            "baseline_cause": bl_pred.predicted_cause,
            "baseline_fix": bl_pred.predicted_fix,
        })

    def accuracy(rows, pred_key, expected_key):
        correct = sum(1 for r in rows if r[pred_key] == r[expected_key])
        return correct / len(rows)

    def hallucination_rate(rows, pred_cause_key, pred_fix_key):
        abstain_rows = [r for r in rows if r["should_abstain"]]
        if not abstain_rows:
            return 0.0
        false_positives = sum(
            1 for r in abstain_rows
            if r[pred_cause_key] is not None or r[pred_fix_key] is not None
        )
        return false_positives / len(abstain_rows)

    metrics = {
        "graphrag": {
            "cause_accuracy": accuracy(rows, "graphrag_cause", "expected_cause"),
            "fix_accuracy": accuracy(rows, "graphrag_fix", "expected_fix"),
            "hallucination_rate": hallucination_rate(rows, "graphrag_cause", "graphrag_fix"),
            "avg_latency_ms": sum(graphrag_latencies) / len(graphrag_latencies),
        },
        "baseline_tfidf": {
            "cause_accuracy": accuracy(rows, "baseline_cause", "expected_cause"),
            "fix_accuracy": accuracy(rows, "baseline_fix", "expected_fix"),
            "hallucination_rate": hallucination_rate(rows, "baseline_cause", "baseline_fix"),
            "avg_latency_ms": sum(baseline_latencies) / len(baseline_latencies),
        },
    }

    print("\n=== Détail par question ===")
    for r in rows:
        print(f'[{r["id"]}] {r["question"]}')
        print(f'    Attendu     -> cause={r["expected_cause"]}  fix={r["expected_fix"]}  (abstention attendue: {r["should_abstain"]})')
        print(f'    GraphRAG    -> cause={r["graphrag_cause"]}  fix={r["graphrag_fix"]}')
        print(f'    Baseline    -> cause={r["baseline_cause"]}  fix={r["baseline_fix"]}')
        print()

    print("=== Résultats agrégés ===")
    print(f'{"Métrique":<22} {"GraphRAG":>12} {"Baseline TF-IDF":>18}')
    print(f'{"Cause accuracy":<22} {metrics["graphrag"]["cause_accuracy"]:>12.0%} {metrics["baseline_tfidf"]["cause_accuracy"]:>18.0%}')
    print(f'{"Fix accuracy":<22} {metrics["graphrag"]["fix_accuracy"]:>12.0%} {metrics["baseline_tfidf"]["fix_accuracy"]:>18.0%}')
    print(f'{"Hallucination rate":<22} {metrics["graphrag"]["hallucination_rate"]:>12.0%} {metrics["baseline_tfidf"]["hallucination_rate"]:>18.0%}')
    print(f'{"Latence moyenne (ms)":<22} {metrics["graphrag"]["avg_latency_ms"]:>12.2f} {metrics["baseline_tfidf"]["avg_latency_ms"]:>18.2f}')

    output_path = ROOT / "eval" / "results.json"
    output_path.write_text(
        json.dumps({"rows": rows, "metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRésultats détaillés sauvegardés dans {output_path}")


if __name__ == "__main__":
    run()
