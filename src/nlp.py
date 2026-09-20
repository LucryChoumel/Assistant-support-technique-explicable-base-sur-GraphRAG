"""
nlp.py
------
Point d'intégration NLP du pipeline — utilisé à l'étape 2 (ancrage d'entités
sur le graphe, dans `kg.find_nodes_by_text`).

Utilise **spaCy** avec le modèle français `fr_core_news_sm` pour :
- lemmatiser les mots de la question (« équipements » -> « équipement »,
  « affichée » -> « afficher »), ce qu'une simple liste de mots-clés ne gère pas ;
- filtrer par catégorie grammaticale (POS tagging) : on ne garde que les noms,
  noms propres, adjectifs et nombres, qui portent l'information utile pour
  l'ancrage (équipement, symptôme, code), et on élimine les mots vides
  ("le", "mon", "après"...) **détectés automatiquement par le modèle**
  (`token.is_stop`) plutôt que via une liste codée en dur.

Le code d'erreur et la version restent extraits par expressions régulières
(`retriever.extract_entities`) : ce sont des motifs strictement numériques,
pour lesquels le NLP n'apporte rien — c'est un choix méthodologique assumé,
pas un oubli.

Si spaCy ou le modèle `fr_core_news_sm` ne sont pas installés (ex :
environnement sans accès à leur téléchargement), le module bascule
automatiquement, de façon transparente, sur un repli par expressions
régulières + liste de mots vides, afin que le projet reste utilisable sans
dépendance obligatoire à un modèle NLP.
"""

from __future__ import annotations

import os
import re

_NLP = None
_NLP_LOAD_ATTEMPTED = False

# Catégories grammaticales jugées porteuses de sens pour l'ancrage d'entités.
# "X" (catégorie indéterminée de spaCy) est inclus car les codes alphanumériques
# de modèles d'équipement (ex: "Y200") y sont fréquemment classés par un
# modèle généraliste non spécialisé sur ce type de vocabulaire technique —
# les exclure a été identifié comme une cause de faux négatifs (voir
# eval/ablation_nlp.py) lors de l'évaluation de ce module.
_RELEVANT_POS = {"NOUN", "PROPN", "ADJ", "NUM", "X"}

# Repli utilisé uniquement si spaCy / le modèle ne sont pas disponibles.
_FALLBACK_STOPWORDS = {
    "the", "and", "pour", "avec", "mon", "ma", "mes", "les", "des",
    "une", "un", "du", "de", "la", "le", "apres", "après", "mise",
    "jour", "affiche", "pourquoi", "equipement", "équipement", "equipment",
}

# Mots vides *spécifiques au domaine*, non détectés comme mots vides
# "génériques" par le modèle de langue (ex: "équipement" est un nom commun
# ordinaire pour spaCy, mais il apparaît dans le label de TOUS les
# équipements du graphe et ne permet donc pas de les distinguer). Appliqué
# dans les deux modes (NLP et repli), en complément — pas en remplacement —
# de la détection de mots vides du modèle.
_DOMAIN_STOPWORDS = {"équipement", "equipement", "equipment", "appareil"}


def _load_spacy_model():
    global _NLP, _NLP_LOAD_ATTEMPTED
    if _NLP_LOAD_ATTEMPTED:
        return _NLP
    _NLP_LOAD_ATTEMPTED = True

    # Interrupteur utilisé pour l'étude d'ablation (eval/ablation_nlp.py) :
    # force le mode de repli sans charger spaCy, afin de mesurer précisément
    # l'apport du modèle NLP par rapport à la méthode qu'il remplace.
    if os.environ.get("FORCE_NLP_FALLBACK"):
        _NLP = None
        return _NLP

    try:
        import spacy

        _NLP = spacy.load("fr_core_news_sm")
    except Exception:
        _NLP = None
    return _NLP


def is_nlp_available() -> bool:
    """Indique si le modèle spaCy a pu être chargé (utilisé par les tests/CLI
    pour afficher explicitement quel mode d'extraction est actif)."""
    return _load_spacy_model() is not None


def extract_keywords(text: str) -> set[str]:
    """Extrait l'ensemble des lemmes significatifs d'un texte.

    Mode NLP (spaCy) : lemmatisation + filtrage grammatical + détection
    automatique des mots vides par le modèle.
    Mode de repli (sans spaCy) : tokenisation regex + liste de mots vides fixe.
    """
    nlp = _load_spacy_model()
    if nlp is not None:
        doc = nlp(text)
        keywords = {
            tok.lemma_.lower()
            for tok in doc
            if tok.pos_ in _RELEVANT_POS and not tok.is_stop and len(tok.lemma_) > 1
        }
    else:
        tokens = re.findall(r"[a-zA-Z0-9àâäéèêëïîôöùûüç]+", text.lower())
        keywords = {t for t in tokens if len(t) > 2 and t not in _FALLBACK_STOPWORDS}

    return keywords - _DOMAIN_STOPWORDS
