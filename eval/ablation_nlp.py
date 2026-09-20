"""
ablation_nlp.py
----------------
Étude d'ablation : compare le pipeline GraphRAG AVEC le modèle NLP
(spaCy `fr_core_news_sm`, lemmatisation + POS tagging) et SANS ce modèle
(repli par expressions régulières + liste de mots vides fixe, méthode
utilisée avant son intégration).

Objectif : mesurer précisément si le modèle NLP choisi apporte un gain
réel sur l'étape qu'il concerne (l'ancrage d'entités), plutôt que de
l'affirmer sans preuve.

Chaque condition est exécutée dans un **sous-processus Python isolé**
(via la variable d'environnement FORCE_NLP_FALLBACK, lue par `src/nlp.py`)
pour garantir qu'aucun état (modèle spaCy déjà chargé en mémoire) ne fuite
d'une condition à l'autre.

Usage :
    python eval/ablation_nlp.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Petit script exécuté dans un sous-processus : reproduit la logique de
# eval/run_evaluation.py mais se contente d'imprimer les métriques en JSON
# sur stdout, pour une lecture programmatique propre par le processus parent.
_WORKER_SCRIPT = r"""
import json, sys, time
sys.path.insert(0, "src")
sys.path.insert(0, "eval")
from kg import KnowledgeGraph
from retriever import GraphRetriever

kg = KnowledgeGraph.from_json("data/sample_kg.json")
retriever = GraphRetriever(kg, max_hops=4, top_k=5)
cases = json.loads(open("eval/eval_dataset.json", encoding="utf-8").read())["cases"]

rows = []
for case in cases:
    result = retriever.retrieve(case["question"])
    cause = fix = None
    for chain in result.chains:
        t = kg.graph.nodes[chain.end_node]["type"]
        if t == "Cause" and cause is None:
            cause = chain.end_node
        if t == "FixProcedure" and fix is None:
            fix = chain.end_node
    rows.append({
        "id": case["id"],
        "expected_cause": case["expected_cause"],
        "expected_fix": case["expected_fix"],
        "should_abstain": case["should_abstain"],
        "predicted_cause": cause,
        "predicted_fix": fix,
        "anchored": result.anchored_entities,
    })

def acc(key_pred, key_exp):
    return sum(1 for r in rows if r[key_pred] == r[key_exp]) / len(rows)

abstain_rows = [r for r in rows if r["should_abstain"]]
false_pos = sum(
    1 for r in abstain_rows
    if r["predicted_cause"] is not None or r["predicted_fix"] is not None
)

print(json.dumps({
    "cause_accuracy": acc("predicted_cause", "expected_cause"),
    "fix_accuracy": acc("predicted_fix", "expected_fix"),
    "hallucination_rate": (false_pos / len(abstain_rows)) if abstain_rows else 0.0,
    "rows": rows,
}))
"""


def run_condition(force_fallback: bool) -> dict:
    env = os.environ.copy()
    if force_fallback:
        env["FORCE_NLP_FALLBACK"] = "1"
    else:
        env.pop("FORCE_NLP_FALLBACK", None)

    result = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def run():
    print("Exécution avec NLP (spaCy fr_core_news_sm)...")
    with_nlp = run_condition(force_fallback=False)

    print("Exécution sans NLP (repli regex + mots vides fixes)...")
    without_nlp = run_condition(force_fallback=True)

    print("\n=== Ablation : impact du modèle NLP sur l'ancrage d'entités ===")
    print(f'{"Métrique":<22} {"Avec spaCy":>12} {"Sans NLP (regex)":>18}')
    print(f'{"Cause accuracy":<22} {with_nlp["cause_accuracy"]:>12.0%} {without_nlp["cause_accuracy"]:>18.0%}')
    print(f'{"Fix accuracy":<22} {with_nlp["fix_accuracy"]:>12.0%} {without_nlp["fix_accuracy"]:>18.0%}')
    print(f'{"Hallucination rate":<22} {with_nlp["hallucination_rate"]:>12.0%} {without_nlp["hallucination_rate"]:>18.0%}')

    print("\n=== Différences d'ancrage question par question ===")
    for r_nlp, r_no in zip(with_nlp["rows"], without_nlp["rows"]):
        if set(r_nlp["anchored"]) != set(r_no["anchored"]):
            print(f'[{r_nlp["id"]}]')
            print(f'   Avec spaCy      -> {r_nlp["anchored"]}')
            print(f'   Sans NLP (regex) -> {r_no["anchored"]}')

    output_path = ROOT / "eval" / "ablation_results.json"
    output_path.write_text(
        json.dumps({"with_nlp": with_nlp, "without_nlp": without_nlp}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRésultats sauvegardés dans {output_path}")


if __name__ == "__main__":
    run()
