from dataclasses import dataclass
from typing import Dict
from mnsr.cognitive_state import CognitiveState
@dataclass
class ControllerConfig:
    """
    Threshold configuration for the Meta-Cognitive Controller.
    """
    confidence_threshold: float = 0.50
    memory_threshold: float = 0.65
    risk_threshold: float = 0.75
    max_steps_per_query: int = 4  
class MetaCognitiveController:
    """
    Meta-Cognitive Controller
    Evaluates the current CognitiveState to determine the next agent action.
    """
    def __init__(self, config: ControllerConfig = ControllerConfig()):
        self.config = config
    def evaluate(self, state: CognitiveState) -> str:
        """
        Evaluates the current state tuple and dictates structural trajectory shifts.
        """
        if state.current_step >= self.config.max_steps_per_query:
            state.set_action("TERMINATE")
            return "TERMINATE"
        if state.contradiction or state.risk_score >= self.config.risk_threshold:
            state.set_action("BACKTRACK")
            return "BACKTRACK"
        if state.confidence < self.config.confidence_threshold:
            state.set_action("REVISE")
            return "REVISE"
        if 0.0 < state.memory_similarity < self.config.memory_threshold:
            state.set_action("RETRIEVE_MEMORY")
            return "RETRIEVE_MEMORY"
        state.set_action("CONTINUE")
        return "CONTINUE"

    def explain(self, state: CognitiveState) -> Dict:
        """
        Returns a diagnostic explanation payload mapping state triggers directly to actions.
        """
        return {
            "chosen_action": state.action,
            "metrics": {
                "confidence": state.confidence,
                "uncertainty": state.uncertainty,
                "contradiction": state.contradiction,
                "memory_similarity": state.memory_similarity,
                "systemic_risk": state.risk_score
            },
            "execution": {
                "step": state.current_step,
                "max_allowed": self.config.max_steps_per_query,
                "loop_pct": round(state.current_step / self.config.max_steps_per_query, 2)
            }
        }