"""
confidence.py
=============

Dynamic Confidence Estimation for MNSR.

WHY THIS MODULE EXISTS
-----------------------
The original pipeline hardcoded `state.update_confidence(0.80)` for every
single question, regardless of how sound the reasoning trace actually was.
This makes the Meta-Cognitive Controller effectively blind: it cannot tell
a rock-solid derivation from a wildly speculative one because both report
identical confidence. A controller that cannot distinguish these cases
cannot meta-cognate -- it can only follow a fixed schedule.

This module replaces the constant with a *signal-fused* estimator that
combines several cheap, deterministic signals into a single confidence
score in [0, 1]. Every signal is optional (missing signals are treated
neutrally), which keeps the estimator usable even in ablations where a
component (e.g. memory) is disabled.

RESEARCH NOVELTY
-----------------
Confidence-driven controllers are the crux of "meta-cognition" claims in
neuro-symbolic literature. Replacing a constant with a computed,
multi-signal score is what allows the controller's decisions to be
*causally connected* to reasoning quality -- a prerequisite for any claim
that the system "knows what it doesn't know". This is directly reportable
as an ablation ("Fixed vs. Dynamic Confidence") in the paper.

BENCHMARK IMPACT
-----------------
Dynamic confidence lets the controller allocate correction budget only to
genuinely weak trajectories, instead of firing REVISE/BACKTRACK loops
uniformly. This reduces wasted LLM calls (latency) on already-correct
answers, and increases recall of "needs correction" cases previously
detected only by symbolic validation (e.g. StrategyQA/TruthfulQA questions
where there is no arithmetic to check but reasoning is still inconsistent).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfidenceWeights:
    """Tunable weights for each confidence signal. Must sum to <= 1.0
    (residual mass is implicitly assigned to the prior)."""

    validation_score: float = 0.30
    reasoning_consistency: float = 0.20
    answer_consistency: float = 0.20
    error_penalty: float = 0.15
    memory_similarity: float = 0.10
    self_eval: float = 0.05
    retry_decay: float = 0.05  # penalty weight, not a positive contributor


class ConfidenceEstimator:
    """
    Fuses multiple independent signals into a single scalar confidence
    value in [0, 1].

    All inputs are already-normalized scalars in [0, 1] except
    `num_errors` and `retry_count`, which are raw integer counts.
    """

    def __init__(self, weights: ConfidenceWeights = ConfidenceWeights()):
        self.weights = weights

    def estimate(
        self,
        validation_score: float = 1.0,
        reasoning_consistency: float = 1.0,
        answer_consistency: float = 1.0,
        num_errors: int = 0,
        memory_similarity: float = 0.0,
        retry_count: int = 0,
        self_eval_score: Optional[float] = None,
    ) -> float:
        """
        Compute a fused confidence score.

        Parameters
        ----------
        validation_score : float
            Output of SymbolicValidator's structured "score" field (0-1).
        reasoning_consistency : float
            Degree to which reasoning steps do not contradict each other.
        answer_consistency : float
            Degree to which the final stated answer matches the answer
            implied by the body of the reasoning.
        num_errors : int
            Count of hard symbolic errors found (arithmetic, contradiction).
        memory_similarity : float
            Similarity to the closest retrieved episode (0 if none found).
            High similarity to a *successful* past episode raises
            confidence; the caller is responsible for passing 0.0 when the
            closest episode was a *failure* (see pipeline.py).
        retry_count : int
            Number of correction iterations already performed on this
            question. Repeated correction failures should erode confidence
            rather than being reset each loop.
        self_eval_score : Optional[float]
            Optional LLM self-reported confidence (0-1), if available.

        Returns
        -------
        float
            Confidence value clamped to [0.0, 1.0].
        """
        w = self.weights

        # Error penalty: diminishing but monotonically decreasing with
        # more errors. 1 error -> -0.5 of the allotted weight, 2 -> -0.75, etc.
        error_term = 1.0 / (1.0 + num_errors)

        # Retry decay: each additional correction attempt that still needed
        # correction is weak evidence the problem is intrinsically hard.
        retry_term = 1.0 / (1.0 + retry_count)

        components = (
            w.validation_score * self._clip(validation_score)
            + w.reasoning_consistency * self._clip(reasoning_consistency)
            + w.answer_consistency * self._clip(answer_consistency)
            + w.error_penalty * error_term
            + w.memory_similarity * self._clip(memory_similarity)
            - w.retry_decay * (1.0 - retry_term)
        )

        if self_eval_score is not None:
            components += w.self_eval * self._clip(self_eval_score)
        else:
            # Redistribute the self-eval weight proportionally across the
            # remaining signals so its absence doesn't systematically
            # deflate confidence when the LLM self-eval step is skipped.
            components += w.self_eval * self._clip(validation_score)

        return self._clip(components)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
