"""
phi3.py
=======

Thin wrapper around a local Ollama Phi-3 Mini model.

WHY THIS CHANGE
---------------
The router's new strategies (SELF_VERIFY, SELF_CRITIQUE, MULTI_PATH_REASONING)
need two capabilities the original wrapper did not expose:

1. A `temperature` override on generation, so MULTI_PATH_REASONING can
   sample several diverse reasoning paths (temperature > 0) while every
   other strategy keeps temperature = 0 for reproducibility.
2. A lightweight `self_evaluate` call that asks the model to rate its own
   confidence in a just-produced answer, feeding the optional
   `self_eval_score` signal in the dynamic confidence estimator.

Both are additive; default behavior (temperature 0, no self-eval) is
unchanged, preserving backward compatibility with any code still calling
`reasoning()`/`generate()` positionally.
"""

import re
from typing import Dict, Any, List, Optional
from ollama import Client


class Phi3Mini:

    def __init__(
        self,
        model: str = "phi3:mini",
        host: str = "http://localhost:11434",
    ):
        self.client = Client(host=host)
        self.model = model

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------
    def extract_answer(self, text: str) -> str:
        if not text:
            return ""

        m = re.search(r"final\s*answer\s*:?\s*(.*)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1]

        return text.strip()

    # ------------------------------------------------------------------
    # Core generation calls
    # ------------------------------------------------------------------
    def _chat(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    def reasoning(self, question: str, temperature: float = 0.0) -> Dict[str, Any]:
        prompt = f"""
Solve the following question carefully.

Question:
{question}

Explain your reasoning step by step.

Finish with exactly:

Final Answer: <answer>
"""
        try:
            text = self._chat(prompt, temperature=temperature)
            return {"reasoning": text, "answer": self.extract_answer(text)}
        except Exception as e:
            return {"reasoning": f"Error: {e}", "answer": ""}

    def generate(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        try:
            text = self._chat(prompt, temperature=temperature)
            return {"text": text, "answer": self.extract_answer(text)}
        except Exception as e:
            return {"text": f"Error: {e}", "answer": ""}

    # ------------------------------------------------------------------
    # New capabilities
    # ------------------------------------------------------------------
    def self_evaluate(self, question: str, reasoning: str, answer: str) -> Optional[float]:
        """
        Asks the model to rate its own confidence in `answer` on a 0-1
        scale, given the question and the reasoning trace that produced
        it. Returns None (rather than raising) on any parse/API failure
        so callers can treat this signal as optional.
        """
        prompt = f"""You previously answered a question. Rate your confidence
that the final answer below is correct, as a single number between 0 and 1
(0 = certainly wrong, 1 = certainly correct). Respond with ONLY the number.

Question:
{question}

Reasoning:
{reasoning}

Final Answer:
{answer}

Confidence (0-1):"""
        try:
            text = self._chat(prompt, temperature=0.0)
            m = re.search(r"(\d*\.?\d+)", text)
            if not m:
                return None
            score = float(m.group(1))
            return max(0.0, min(1.0, score))
        except Exception:
            return None

    def generate_multi_path(
        self, question: str, num_paths: int = 3, temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Samples `num_paths` diverse reasoning trajectories for
        majority-vote / multi-path strategies."""
        paths = []
        for _ in range(max(1, num_paths)):
            paths.append(self.reasoning(question, temperature=temperature))
        return paths

    # ------------------------------------------------------------------
    # Batch helpers (kept for backward compatibility)
    # ------------------------------------------------------------------
    def process_single_item(self, question: str):
        return self.reasoning(question)

    def generate_all_parallel(self, questions: List[str]):
        return [self.reasoning(q) for q in questions]
