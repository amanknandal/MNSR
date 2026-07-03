from typing import Dict
from models.phi3 import Phi3Mini
from mnsr.symbolic_validator import SymbolicValidator
from mnsr.cognitive_state import CognitiveState
from mnsr.controller import MetaCognitiveController
from mnsr.router import StrategyRouter
from mnsr.memory import ReflectionMemory

class MNSRPipeline:
    """
    Complete Meta-Cognitive Neuro-Symbolic Reasoning Pipeline.
    Executes iterative correction steps controlled dynamically by CognitiveState updates.
    """

    def __init__(self):
        self.model = Phi3Mini()
        self.validator = SymbolicValidator()
        self.controller = MetaCognitiveController()
        self.router = StrategyRouter(self.model)
        self.memory = ReflectionMemory()
    def solve(self, question: str) -> Dict:
        state = CognitiveState()
        result = self.model.reasoning(question)
        state.update_reasoning(result["reasoning"])
        state.set_answer(result["answer"])
        state.update_confidence(0.80)  
        
        memory_hint = ""
        while True:
            state.next_step()
            report = self.validator.validate(state.reasoning)
            state.update_symbolic_result(report)
            retrieved = self.memory.retrieve(question)
            if retrieved is not None:
                state.update_memory_similarity(retrieved["similarity"])
                memory_hint = retrieved["episode"]["corrected_reasoning"]
            else:
                state.update_memory_similarity(0.0)
            action = self.controller.evaluate(state)
            if action in ("CONTINUE", "TERMINATE"):
                break
            updated_execution = self.router.execute(
                action=action,
                question=question,
                state=state,
                memory_hint=memory_hint
            )
            state.update_reasoning(updated_execution["reasoning"])
            state.set_answer(updated_execution["answer"])
        final_report = self.validator.validate(state.reasoning)
        if final_report["valid"]:
            self.memory.add(
                question=question,
                reasoning=result["reasoning"], 
                answer=state.final_answer,
                errors=final_report["errors"],
                corrected_reasoning=state.reasoning, 
                success=True
            )

        return {
            "question": question,
            "reasoning": state.reasoning,
            "answer": state.final_answer,
            "final_action": action,
            "validation": final_report,
            "state": state.to_dict()
        }