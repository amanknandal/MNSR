from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class CognitiveState:
    """
    Represents the current reasoning state of the MNSR agent.
    Maintains the exact mathematical tuple formalization: s_t = (h_t, c_t, u_t, k_t, m_t)
    """
    reasoning: str = ""              
    confidence: float = 1.0         
    uncertainty: float = 0.0        
    contradiction: bool = False     
    memory_similarity: float = 0.0   
    current_step: int = 0
    total_steps: int = 0
    risk_score: float = 0.0          
    symbolic_errors: List[Dict] = field(default_factory=list)
    action: Optional[str] = None
    final_answer: Optional[str] = None
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
        self.contradiction = not report.get("valid", True)
        self._calculate_derived_risk()
    def update_memory_similarity(self, similarity: float):
        self.memory_similarity = max(0.0, min(1.0, similarity))
    def set_action(self, action: str):
        self.action = action
    def set_answer(self, answer: str):
        self.final_answer = answer
    def next_step(self):
        self.current_step += 1
    def _calculate_derived_risk(self):
        """
        Dynamically calculates systemic risk state based on uncertainty 
        and the presence of symbolic contradictions.
        """
        base_risk = self.uncertainty
        if self.contradiction:
            base_risk = max(base_risk, 0.7) + (0.1 * len(self.symbolic_errors))
        self.risk_score = max(0.0, min(1.0, round(base_risk, 4)))
    def reset(self):
        """Resets state components to prevent inter-evaluation sample leakage."""
        self.reasoning = ""
        self.confidence = 1.0
        self.uncertainty = 0.0
        self.contradiction = False
        self.memory_similarity = 0.0
        self.current_step = 0
        self.total_steps = 0
        self.risk_score = 0.0
        self.symbolic_errors = [] 
        self.action = None
        self.final_answer = None
    def to_dict(self) -> Dict:
        return {
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "contradiction": self.contradiction,
            "memory_similarity": self.memory_similarity,
            "risk_score": self.risk_score,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "action": self.action,
            "answer": self.final_answer,
            "symbolic_errors": self.symbolic_errors
        }
    def __repr__(self) -> str:
        return (f"CognitiveState(step={self.current_step}, confidence={self.confidence}, "
                f"contradiction={self.contradiction}, risk={self.risk_score}, action='{self.action}')")