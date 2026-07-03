from ollama import Client
from typing import Dict, Any
import re
class Phi3Mini:
    """
    Wrapper for Phi-3 Mini running through Ollama.
    """
    def __init__(
        self,
        model: str = "phi3:mini",
        host: str = "http://localhost:11434"
    ):
        self.client = Client(host=host)
        self.model = model
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": temperature
            }
        )
        text = response["message"]["content"]
        return {
            "text": text,
            "raw": response
        }
    def extract_answer(self, response_text: str):
        numbers = re.findall(r"-?\d+\.?\d*", response_text)
        if len(numbers) == 0:
            return None
        return numbers[-1]
    def reasoning(self, question: str):
        prompt = f"""
Solve the following problem step by step.
Question:
{question}
After your reasoning, end with:
Final Answer: <answer>
"""
        result = self.generate(prompt)
        answer = self.extract_answer(result["text"])
        return {
            "reasoning": result["text"],
            "answer": answer
        }