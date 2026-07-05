"""
memory.py
=========

Episodic Reflection Memory for MNSR (success + failure episodes).

WHY THIS CHANGE
---------------
The original memory only stored episodes where final validation passed,
and only ever returned a single closest match with no notion of whether
that match was itself a success or a failure. This means the controller
could never distinguish "I've solved something like this before" from
"I've *failed* at something like this before" -- both of which are
useful but imply opposite corrective strategies (reuse vs. avoid).

This version stores both outcome types in a single TF-IDF index (kept
lightweight and dependency-minimal, consistent with the original design),
tags each episode with its outcome, and returns the best match *per
outcome type* rather than a single global best match, plus the fields
required by the improved spec: embedding (TF-IDF vector, exposed via
`export_vector`), corrected reasoning, controller actions taken, retry
count, and confidence.

RESEARCH NOVELTY
-----------------
Failure-memory retrieval is the mechanism that lets MNSR claim genuine
error-avoidance rather than error-correction only -- the system can
recognize "I have been wrong on a question like this before" *before*
generating a new answer, not just after. This is the basis for the
MEMORY_REFLECTION controller action and is directly reportable as an
ablation ("- Failure Memory") separate from the existing "- Memory"
ablation.

BENCHMARK IMPACT
-----------------
On datasets with repeated question *patterns* (e.g. StrategyQA questions
sharing a reasoning template, GSM8K word-problem archetypes), failure
memory should reduce repeat mistakes across the evaluation set, which
compounds as the run progresses (accuracy on later batches should exceed
accuracy on earlier batches within the same run -- a testable, reportable
trend).
"""

import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ReflectionMemory:
    """
    Episodic Reflection Memory.

    Stores both successful and failed reasoning episodes and uses
    TF-IDF + cosine similarity to retrieve the closest precedent of each
    outcome type for a new question.
    """

    def __init__(self):
        self.memory: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def add(
        self,
        question: str,
        reasoning: str,
        answer: str,
        errors: List[Dict],
        corrected_reasoning: str = "",
        success: bool = False,
        controller_actions: Optional[List[str]] = None,
        retry_count: int = 0,
        confidence: float = 0.0,
        validation_report: Optional[Dict] = None,
    ):
        episode = {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "errors": errors,
            "corrected_reasoning": corrected_reasoning,
            "success": success,
            "controller_actions": controller_actions or [],
            "retry_count": retry_count,
            "confidence": confidence,
            "validation_report": validation_report or {},
        }
        self.memory.append(episode)
        self._rebuild_index()

    def _rebuild_index(self):
        questions = [ep["question"].lower() for ep in self.memory]
        # TF-IDF requires at least one non-empty vocabulary; guard against
        # a memory of only empty-string questions.
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(questions)
        except ValueError:
            self.tfidf_matrix = None

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def retrieve(self, question: str, threshold: float = 0.60) -> Dict[str, Optional[Dict]]:
        """
        Returns a dict:
            {
                "success": {"similarity": float, "episode": {...}} or None,
                "failure": {"similarity": float, "episode": {...}} or None,
            }
        Each entry is None if no episode of that outcome type meets the
        similarity threshold.
        """
        result = {"success": None, "failure": None}
        if not self.memory or self.tfidf_matrix is None:
            return result

        query_vec = self.vectorizer.transform([question.lower()])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        for outcome_key, want_success in (("success", True), ("failure", False)):
            candidate_indices = [
                i for i, ep in enumerate(self.memory) if ep["success"] == want_success
            ]
            if not candidate_indices:
                continue
            best_idx = max(candidate_indices, key=lambda i: similarities[i])
            best_score = float(similarities[best_idx])
            if best_score >= threshold:
                result[outcome_key] = {
                    "similarity": round(best_score, 3),
                    "episode": self.memory[best_idx],
                }
        return result

    def retrieve_best(self, question: str, threshold: float = 0.60) -> Optional[Dict]:
        """Backward-compatible single-best-match retrieval (any outcome),
        matching the original API used by earlier pipeline versions."""
        both = self.retrieve(question, threshold=threshold)
        candidates = [v for v in both.values() if v is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c["similarity"])

    def size(self) -> int:
        return len(self.memory)

    def success_count(self) -> int:
        return sum(1 for ep in self.memory if ep["success"])

    def failure_count(self) -> int:
        return sum(1 for ep in self.memory if not ep["success"])

    def clear(self):
        self.memory = []
        self.tfidf_matrix = None

    def export(self) -> List[Dict]:
        return self.memory
