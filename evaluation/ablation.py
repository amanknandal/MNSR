import copy
import json
from typing import Dict, Any
from evaluation.experiment import MNSRExperiment
class MockValidator:
    def validate(self, reasoning: str) -> Dict[str, Any]:
        return {"valid": True, "errors": [], "num_errors": 0}
class MockMemory:
    def retrieve(self, question: str, threshold: float = 0.60) -> None:
        return None
    def add(self, *args, **kwargs):
        pass
class MockController:
    def evaluate(self, state) -> str:
        return "CONTINUE"
class AblationStudy:
    """
    Performs component-wise ablation experiments on MNSR.
    Safely bypasses component classes using structural null mocks to evaluate system degradation.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.results: Dict[str, Dict[str, Any]] = {}
    def _run_pipeline(
        self,
        disable_validator: bool = False,
        disable_memory: bool = False,
        disable_controller: bool = False
    ) -> Dict[str, Any]:   
        experiment = MNSRExperiment(self.dataset_path)
        if disable_validator:
            experiment.mnsr.validator = MockValidator()
        if disable_memory:
            experiment.mnsr.memory = MockMemory()
        if disable_controller:
            experiment.mnsr.controller = MockController()
        experiment.run()
        return copy.deepcopy(experiment.metrics.summary())
    def run(self) -> Dict[str, Dict[str, Any]]:
        print("\n[Ablation] Running Full MNSR Baseline Configuration...")
        self.results["Full MNSR"] = self._run_pipeline()
        print("\n[Ablation] Running Configuration: (- Symbolic Validator)...")
        self.results["- Validator"] = self._run_pipeline(disable_validator=True)
        print("\n[Ablation] Running Configuration: (- Reflection Memory Index)...")
        self.results["- Memory"] = self._run_pipeline(disable_memory=True)
        print("\n[Ablation] Running Configuration: (- Meta-Cognitive Controller)...")
        self.results["- Controller"] = self._run_pipeline(disable_controller=True)
        return self.results
    def print_report(self):
        print("\n" + "="*16 + " ABLATION STUDY ANALYSIS REPORT " + "="*16 + "\n")
        print(f"{'System Configuration':28} {'Accuracy':>12} {'Avg Steps':>12}")
        print("-" * 55)
        for name, metrics in self.results.items():
            acc = metrics.get("mnsr_accuracy", 0.0)
            steps = metrics.get("avg_steps", 0.0)
            print(f"{name:28} {acc * 100:>11.2f}% {steps:>12.2f}")
        print("\n" + "="*64 + "\n")
    def save(self, filename="ablation_results.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        print(f"\nAblation study metrics data matrix exported -> {filename}")