"""
cognitive_state.py
===================

Expanded CognitiveState for MNSR.

WHY THIS CHANGE
---------------
The original state tuple s_t = (h_t, c_t, u_t, k_t, m_t) only tracked
reasoning text, confidence, uncertainty, contradiction flag, and memory
similarity. This is too coarse for the richer controller/router described
in the improved architecture: the controller now needs answer/reasoning
*consistency* scores, a *correction history* (which strategies were
already tried, to avoid infinite REVISE<->BACKTRACK oscillation), a
*retry count* independent of `current_step` (so different modules can
reset step counters without losing the global retry budget), and timing
information for latency/overhead reporting.

RESEARCH NOVELTY
-----------------
Explicitly modeling `correction_history` lets the controller apply a
"no-repeat" rule (a form of tabu search) so the same failed strategy is
not retried verbatim -- a common failure mode in naive self-refinement
loops reported in the self-correction literature (e.g. "LLMs cannot
self-correct reasoning yet"). Tracking this is what allows MNSR to claim
it avoids that specific known failure mode.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class CognitiveState:
    """
    Represents the current reasoning state of the MNSR agent.

    Extended tuple: s_t = (h_t, c_t, u_t, k_t, m_t, a_t, r_t, hist_t, n_t)
    where:
        h_t    -- reasoning trace
        c_t    -- confidence
        u_t    -- uncertainty (1 - confidence, kept separate for clarity)
        k_t    -- contradiction / symbolic validity
        m_t    -- memory similarity
        a_t    -- answer & answer/reasoning consistency
        r_t    -- risk score (derived)
        hist_t -- correction history (strategies already attempted)
        n_t    -- retry count / step count
    """

    # --- reasoning & answer ---
    reasoning: str = ""
    final_answer: Optional[str] = None

    # --- confidence family ---
    confidence: float = 1.0
    uncertainty: float = 0.0

    # --- symbolic validation family ---
    contradiction: bool = False
    validation_score: float = 1.0
    symbolic_errors: List[Dict] = field(default_factory=list)
    symbolic_warnings: List[Dict] = field(default_factory=list)

    # --- consistency signals ---
    reasoning_consistency: float = 1.0
    answer_consistency: float = 1.0

    # --- memory ---
    memory_similarity: float = 0.0
    memory_match_type: Optional[str] = None  # "success" | "failure" | None

    # --- risk / control ---
    risk_score: float = 0.0
    action: Optional[str] = None
    correction_history: List[str] = field(default_factory=list)

    # --- iteration bookkeeping ---
    current_step: int = 0
    total_steps: int = 0
    retry_count: int = 0
    max_reflection_depth: int = 4

    # --- timing ---
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------
    def update_reasoning(self, reasoning: str):
        self.reasoning = reasoning
        self.total_steps = len([line for line in reasoning.split("\n") if line.strip()])
        self._calculate_derived_risk()

    def update_confidence(self, confidence: float):
        self.confidence = max(0.0, min(1.0, confidence))
        self.uncertainty = round(1.0 - self.confidence, 4)
        self._calculate_derived_risk()

    def update_symbolic_result(self, report: Dict):
        self.symbolic_errors = report.get("errors", [])
        self.symbolic_warnings = report.get("warnings", [])
        self.validation_score = report.get("score", 1.0 if report.get("valid", True) else 0.0)
        self.reasoning_consistency = report.get("reasoning_consistency", self.reasoning_consistency)
        self.answer_consistency = report.get("answer_consistency", self.answer_consistency)
        self.contradiction = not report.get("valid", True)
        self._calculate_derived_risk()

    def update_memory_similarity(self, similarity: float, match_type: Optional[str] = None):
        self.memory_similarity = max(0.0, min(1.0, similarity))
        self.memory_match_type = match_type

    def set_action(self, action: str):
        self.action = action
        if action not in ("CONTINUE", "TERMINATE"):
            self.correction_history.append(action)

    def set_answer(self, answer: str):
        self.final_answer = answer

    def next_step(self):
        self.current_step += 1
        self.elapsed_time = round(time.time() - self.start_time, 4)

    def increment_retry(self):
        self.retry_count += 1

    def was_recently_attempted(self, action: str, window: int = 2) -> bool:
        """Tabu-style check: has `action` already been tried in the last
        `window` correction steps? Used by the controller to avoid
        oscillating between the same two failing strategies."""
        return action in self.correction_history[-window:]

    def _calculate_derived_risk(self):
        """
        Systemic risk fuses uncertainty, symbolic contradiction severity,
        and consistency shortfalls into a single scalar used by the
        controller's BACKTRACK/REPLAN gate.
        """
        base_risk = self.uncertainty
        inconsistency_penalty = (1.0 - self.reasoning_consistency) * 0.3 + (
            1.0 - self.answer_consistency
        ) * 0.3
        base_risk = max(base_risk, inconsistency_penalty)
        if self.contradiction:
            base_risk = max(base_risk, 0.7) + (0.1 * len(self.symbolic_errors))
        self.risk_score = max(0.0, min(1.0, round(base_risk, 4)))

    def reset(self):
        """Resets state components to prevent inter-evaluation sample leakage."""
        self.reasoning = ""
        self.final_answer = None
        self.confidence = 1.0
        self.uncertainty = 0.0
        self.contradiction = False
        self.validation_score = 1.0
        self.symbolic_errors = []
        self.symbolic_warnings = []
        self.reasoning_consistency = 1.0
        self.answer_consistency = 1.0
        self.memory_similarity = 0.0
        self.memory_match_type = None
        self.risk_score = 0.0
        self.action = None
        self.correction_history = []
        self.current_step = 0
        self.total_steps = 0
        self.retry_count = 0
        self.start_time = time.time()
        self.elapsed_time = 0.0

    def to_dict(self) -> Dict:
        return {
            "reasoning": self.reasoning,
            "answer": self.final_answer,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "contradiction": self.contradiction,
            "validation_score": self.validation_score,
            "reasoning_consistency": self.reasoning_consistency,
            "answer_consistency": self.answer_consistency,
            "memory_similarity": self.memory_similarity,
            "memory_match_type": self.memory_match_type,
            "risk_score": self.risk_score,
            "action": self.action,
            "correction_history": self.correction_history,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "retry_count": self.retry_count,
            "elapsed_time": self.elapsed_time,
            "symbolic_errors": self.symbolic_errors,
            "symbolic_warnings": self.symbolic_warnings,
        }

    def __repr__(self) -> str:
        return (
            f"CognitiveState(step={self.current_step}, confidence={self.confidence}, "
            f"contradiction={self.contradiction}, risk={self.risk_score}, "
            f"action='{self.action}', retries={self.retry_count})"
        )
