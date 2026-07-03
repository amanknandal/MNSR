from typing import Dict
from mnsr.cognitive_state import CognitiveState
class StrategyRouter:
    """
    Executes the action selected by the Meta-Cognitive Controller.
    The router never decides. It only executes.
    """
    def __init__(self, model):
        self.model = model
    def execute(
        self,
        action: str,
        question: str,
        state: CognitiveState,
        memory_hint: str = ""
    ) -> Dict:
        if action in ("CONTINUE", "TERMINATE"):
            return {
                "reasoning": state.reasoning,
                "answer": state.final_answer
            }
        elif action == "REVISE":
            prompt = f"""You are reviewing a mathematical/logical problem solver. The following reasoning may contain mistakes. Review it, correct any bad logic, and rewrite only the flawed steps.
Question:
{question}

Previous reasoning trail:
{state.reasoning}
Provide your corrected step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        elif action == "BACKTRACK":
            prompt = f"""The previous reasoning trail contained fatal symbolic contradictions or arithmetic errors. Discard the failed thinking strategy entirely and solve the question completely fresh from the beginning.
Question:
{question}

Provide your completely new step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        elif action == "RETRIEVE_MEMORY":
            prompt = f"""You have access to a reference memory lookup showing a past similar mistake and its correction. Use this contextual hint to avoid repeating the error on the current target problem.
Reference Memory Hint:
{memory_hint}
Target Question:
{question}
Provide your step-by-step reasoning using the reference knowledge.
After your reasoning, end with:
Final Answer: <answer>"""
        else:
            raise ValueError(f"Unknown system execution action: {action}")
        result = self.model.generate(prompt)
        extracted_ans = self.model.extract_answer(result["text"])

        return {
            "reasoning": result["text"],
            "answer": extracted_ans
        }