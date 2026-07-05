import re
from typing import Dict, Any, List
from ollama import Client


class Phi3Mini:

    def __init__(
        self,
        model: str = "phi3:mini",
        host: str = "http://localhost:11434"
    ):
        self.client = Client(host=host)
        self.model = model

    def extract_answer(self, text: str) -> str:
        if not text:
            return ""

        m = re.search(
            r"final\s*answer\s*:?\s*(.*)",
            text,
            re.IGNORECASE
        )

        if m:
            return m.group(1).strip()

        numbers = re.findall(r"-?\d+\.?\d*", text)

        if numbers:
            return numbers[-1]

        return text.strip()

    def reasoning(self, question: str) -> Dict[str, Any]:
        prompt = f"""
Solve the following question carefully.

Question:
{question}

Explain your reasoning step by step.

Finish with exactly:

Final Answer: <answer>
"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0
                }
            )

            text = response["message"]["content"]

            return {
                "reasoning": text,
                "answer": self.extract_answer(text)
            }

        except Exception as e:
            return {
                "reasoning": f"Error: {e}",
                "answer": ""
            }

    def generate(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0
                }
            )

            text = response["message"]["content"]

            return {
                "text": text,
                "answer": self.extract_answer(text)
            }

        except Exception as e:
            return {
                "text": f"Error: {e}",
                "answer": ""
            }

    def process_single_item(self, question: str):
        return self.reasoning(question)

    def generate_all_parallel(self, questions: List[str]):
        return [self.reasoning(q) for q in questions]