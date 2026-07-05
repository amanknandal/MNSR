"""
controller.py
==============

Meta-Cognitive Controller for MNSR (expanded action space).

WHY THIS CHANGE
---------------
The original controller only had four possible outputs (CONTINUE,
TERMINATE, REVISE, BACKTRACK, RETRIEVE_MEMORY) driven almost entirely by
`confidence` and `contradiction`. That collapses many qualitatively
different failure modes (a wrong final number vs. an unsupported logical
leap vs. a genuinely under-specified question) into the same one or two
corrective actions. This version widens the action space to match
distinct repair strategies in router.py, and bases the decision on a
richer state (validation score, answer/reasoning consistency, memory
similarity + match type, correction history, reasoning length, retry
count, uncertainty) rather than confidence alone.

The decision function is a **strict, ordered priority list** -- this is
intentional and documented below so the controller remains deterministic
and reproducible (required by the "Research Quality" section): given an
identical CognitiveState, `evaluate()` always returns the same action.

RESEARCH NOVELTY
-----------------
A controller with a differentiated action space lets the paper report
*which* corrective strategy fires for *which* failure signature (e.g.
"DECOMPOSE fires for long, low reasoning-consistency traces" or
"ANSWER_REPAIR fires for answer/reasoning mismatches with otherwise sound
logic"), which is a much stronger and more interpretable claim than "the
system revised N times."

BENCHMARK IMPACT
-----------------
Precisely targeted repairs (e.g. ANSWER_REPAIR only touches the final
answer extraction instead of regenerating the entire reasoning trace)
reduce unnecessary LLM calls relative to always doing a full REVISE/
BACKTRACK, which is expected to lower average latency while preserving or
improving accuracy gains, since low-risk failures no longer force a full
rewrite.
"""

from dataclasses import dataclass
from typing import Dict
from mnsr.cognitive_state import CognitiveState


@dataclass
class ControllerConfig:
    """Threshold configuration for the Meta-Cognitive Controller."""

    confidence_threshold: float = 0.55
    validation_threshold: float = 0.60
    answer_consistency_threshold: float = 0.60
    reasoning_consistency_threshold: float = 0.60
    memory_threshold: float = 0.65
    risk_threshold: float = 0.75
    max_reasoning_lines_for_decompose: int = 3
    max_steps_per_query: int = 4
    max_retry_count: int = 6


class MetaCognitiveController:
    """
    Evaluates the current CognitiveState and deterministically selects the
    next agent action.

    DECISION ORDER (checked top to bottom, first match wins):
    ------------------------------------------------------------------
    1. TERMINATE            -- step or retry budget exhausted.
    2. BACKTRACK             -- severe systemic risk (contradiction AND
                                 risk over threshold) not already retried.
    3. REPLAN                -- same severe-risk condition, but BACKTRACK
                                 was already tried this trajectory (avoid
                                 oscillation / tabu).
    4. ANSWER_REPAIR         -- reasoning is validated & consistent but the
                                 stated final answer disagrees with it.
    5. REASONING_REPAIR      -- hard symbolic errors present but no
                                 systemic contradiction.
    6. DECOMPOSE             -- reasoning is too short/shallow for the
                                 apparent problem (few lines, low
                                 confidence) -- likely needs sub-questions.
    7. MEMORY_REFLECTION     -- a similar *failed* past episode was
                                 retrieved above the memory threshold.
    8. SELF_VERIFY           -- confidence borderline, validation OK:
                                 ask the model to double-check itself
                                 rather than fully regenerate.
    9. SELF_CRITIQUE         -- reasoning consistency low but no hard
                                 error was found (soft signal).
    10. MULTI_PATH_REASONING -- confidence still below threshold after
                                 at least one prior correction attempt
                                 (single-path repair isn't converging).
    11. CONTINUE             -- default: state is healthy enough to accept.
    ------------------------------------------------------------------
    """

    def __init__(self, config: ControllerConfig = ControllerConfig()):
        self.config = config

    def evaluate(self, state: CognitiveState) -> str:
        cfg = self.config

        # 1. Hard budget stop.
        if state.current_step >= cfg.max_steps_per_query or state.retry_count >= cfg.max_retry_count:
            return self._select(state, "TERMINATE")

        severe_risk = state.contradiction or state.risk_score >= cfg.risk_threshold

        # 2 / 3. Systemic risk -> BACKTRACK, or REPLAN if BACKTRACK already tried.
        if severe_risk:
            if state.was_recently_attempted("BACKTRACK", window=len(state.correction_history)):
                return self._select(state, "REPLAN")
            return self._select(state, "BACKTRACK")

        # 4. Sound reasoning, inconsistent final answer -> cheap targeted repair.
        if (
            state.validation_score >= cfg.validation_threshold
            and state.reasoning_consistency >= cfg.reasoning_consistency_threshold
            and state.answer_consistency < cfg.answer_consistency_threshold
        ):
            return self._select(state, "ANSWER_REPAIR")

        # 5. Hard symbolic errors without full contradiction.
        if state.validation_score < cfg.validation_threshold and len(state.symbolic_errors) > 0:
            return self._select(state, "REASONING_REPAIR")

        # 6. Reasoning too shallow to trust.
        if (
            state.total_steps <= cfg.max_reasoning_lines_for_decompose
            and state.confidence < cfg.confidence_threshold
        ):
            return self._select(state, "DECOMPOSE")

        # 7. Relevant failure precedent retrieved from memory.
        if (
            state.memory_match_type == "failure"
            and state.memory_similarity >= cfg.memory_threshold
        ):
            return self._select(state, "MEMORY_REFLECTION")

        # 8. Borderline confidence but otherwise clean -> lightweight self-check.
        if (
            state.confidence < cfg.confidence_threshold
            and state.validation_score >= cfg.validation_threshold
            and state.retry_count == 0
        ):
            return self._select(state, "SELF_VERIFY")

        # 9. Soft internal inconsistency without a hard error.
        if state.reasoning_consistency < cfg.reasoning_consistency_threshold:
            return self._select(state, "SELF_CRITIQUE")

        # 10. Confidence still low after at least one correction attempt.
        if state.confidence < cfg.confidence_threshold and state.retry_count >= 1:
            return self._select(state, "MULTI_PATH_REASONING")

        # 11. Healthy state.
        return self._select(state, "CONTINUE")

    @staticmethod
    def _select(state: CognitiveState, action: str) -> str:
        state.set_action(action)
        return action

    def explain(self, state: CognitiveState) -> Dict:
        """Diagnostic payload mapping state triggers directly to actions."""
        return {
            "chosen_action": state.action,
            "metrics": {
                "confidence": state.confidence,
                "uncertainty": state.uncertainty,
                "validation_score": state.validation_score,
                "reasoning_consistency": state.reasoning_consistency,
                "answer_consistency": state.answer_consistency,
                "contradiction": state.contradiction,
                "memory_similarity": state.memory_similarity,
                "memory_match_type": state.memory_match_type,
                "systemic_risk": state.risk_score,
            },
            "execution": {
                "step": state.current_step,
                "retry_count": state.retry_count,
                "max_allowed_steps": self.config.max_steps_per_query,
                "loop_pct": round(state.current_step / self.config.max_steps_per_query, 2),
                "correction_history": state.correction_history,
            },
        }
