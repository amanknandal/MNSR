from dataclasses import dataclass, asdict
from typing import Dict
@dataclass
class ExperimentMetrics:
    """
    Tracks all metrics used in MNSR evaluation.
    """
    total_questions: int = 0
    baseline_correct: int = 0
    mnsr_correct: int = 0
    arithmetic_errors_detected: int = 0
    arithmetic_errors_fixed: int = 0
    revisions: int = 0
    backtracks: int = 0
    memory_retrievals: int = 0
    total_steps: int = 0
    def update(
        self,
        baseline_correct: bool,
        mnsr_correct: bool,
        validation_report: Dict,
        final_action: str,
        steps: int
    ):
        self.total_questions += 1
        self.total_steps += steps
        if baseline_correct:
            self.baseline_correct += 1
        if mnsr_correct:
            self.mnsr_correct += 1
        num_errors = validation_report.get("num_errors", 0)
        if num_errors > 0:
            self.arithmetic_errors_detected += num_errors

        if num_errors > 0 and mnsr_correct and not baseline_correct:
            self.arithmetic_errors_fixed += 1
        if final_action == "REVISE":
            self.revisions += 1
        elif final_action == "BACKTRACK":
            self.backtracks += 1
        elif final_action == "RETRIEVE_MEMORY":
            self.memory_retrievals += 1
    def summary(self) -> Dict:
        if self.total_questions == 0:
            return {}
        return {
            "baseline_accuracy": round(self.baseline_correct / self.total_questions, 4),
            "mnsr_accuracy": round(self.mnsr_correct / self.total_questions, 4),
            "error_detection_rate": round(self.arithmetic_errors_detected / self.total_questions, 4),
            "error_fix_rate": round(
                self.arithmetic_errors_fixed / max(1, self.arithmetic_errors_detected), 4
            ),
            "avg_steps": round(self.total_steps / self.total_questions, 2),
            "revisions": self.revisions,
            "backtracks": self.backtracks,
            "memory_retrievals": self.memory_retrievals
        }
    def export(self) -> Dict:
        return asdict(self)