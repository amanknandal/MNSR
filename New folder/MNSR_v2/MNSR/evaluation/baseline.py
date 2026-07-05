"""
baseline.py
===========

WHY THIS CHANGE
---------------
Baselines now accept the same `dataset_hint` parameter as
`MNSRPipeline.solve()` so `ValidatorBaseline`/`RevisionBaseline` validate
against the correct dataset-specific format constraints (spec item 2 /
item 9), keeping the baseline vs. MNSR comparison apples-to-apples rather
than validating MNSR against dataset-aware rules while the baseline is
checked against dataset-agnostic ones.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.phi3 import Phi3Mini
from mnsr.symbolic_validator import SymbolicValidator
from mnsr.pipeline import MNSRPipeline


class BaseReasoner(ABC):
    """Abstract base class for all baseline inference architectures."""

    def __init__(self):
        self.model = Phi3Mini()

    @abstractmethod
    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        pass


class BaselineCoT(BaseReasoner):
    """
    Standard Phi-3 Chain-of-Thought.
    No symbolic validation, feedback correction loop, or long-term memory.
    """

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        result = self.model.reasoning(question)
        return {
            "method": "ChainOfThought",
            "question": question,
            "reasoning": result.get("reasoning", ""),
            "answer": result.get("answer", ""),
        }


class ValidatorBaseline(BaseReasoner):
    """
    Runs symbolic validation exactly once after text extraction.
    Flags errors but does not attempt any system recovery strategies.
    """

    def __init__(self):
        super().__init__()
        self.validator = SymbolicValidator()

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        dataset_type = MNSRPipeline.infer_dataset_type(question, dataset_hint)
        result = self.model.reasoning(question)
        report = self.validator.validate(result.get("reasoning", ""), dataset_type=dataset_type)
        return {
            "method": "ValidatorOnly",
            "question": question,
            "reasoning": result.get("reasoning", ""),
            "answer": result.get("answer", ""),
            "validation": report,
        }


class RevisionBaseline(BaseReasoner):
    """
    Performs exactly one symbolic validation pass.
    If an error is discovered, runs a single unguided generation pass to
    attempt a fix (no meta-cognitive routing, for ablation comparison
    against the full MNSR strategy router).
    """

    def __init__(self):
        super().__init__()
        self.validator = SymbolicValidator()

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        dataset_type = MNSRPipeline.infer_dataset_type(question, dataset_hint)
        result = self.model.reasoning(question)

        reasoning = result.get("reasoning", "")
        answer = result.get("answer", "")

        report = self.validator.validate(reasoning, dataset_type=dataset_type)

        if not report.get("valid", True):
            prompt = f"""The following reasoning contains arithmetic or logical errors.
Question:
{question}
Previous reasoning:
{reasoning}
Please correct the reasoning.
Finish with:
Final Answer: <answer>"""
            revised = self.model.generate(prompt)
            if isinstance(revised, dict):
                reasoning = revised.get("text", revised.get("generation", revised.get("reasoning", "")))
            else:
                reasoning = str(revised)
            if hasattr(self.model, "extract_answer"):
                answer = self.model.extract_answer(reasoning)
            elif isinstance(revised, dict) and "answer" in revised:
                answer = revised["answer"]

        return {
            "method": "ValidatorRevision",
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "validation": report,
        }


def get_all_baselines() -> List[BaseReasoner]:
    return [BaselineCoT(), ValidatorBaseline(), RevisionBaseline()]
