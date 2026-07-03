import re
from typing import Dict, List
class SymbolicValidator:
    """
    Symbolic validation layer for MNSR.
    Version 1:
        - arithmetic checking
        - expandable to logic/rules later
    """
    def __init__(self):
        self.errors = []
    def validate(self, reasoning: str) -> Dict:
        self.errors = []
        self.arithmetic_check(reasoning)
        self.logic_check(reasoning)
        self.contradiction_check(reasoning)
        return self.final_report()
    def arithmetic_check(self, reasoning: str):
        pattern = r'(-?\d+\.?\d*)\s*([\+\-\*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)'
        matches = re.finditer(pattern, reasoning)
        for m in matches:
            a = float(m.group(1))
            op = m.group(2)
            b = float(m.group(3))
            claimed = float(m.group(4))
            
            if op == "+":
                actual = a + b
            elif op == "-":
                actual = a - b
            elif op == "*":
                actual = a * b
            elif op == "/":
                if b == 0:
                    continue
                actual = a / b
            else:
                continue
            if abs(actual - claimed) > 1e-6:
                self.errors.append({
                    "type": "Arithmetic Error",
                    "expression": m.group(0),
                    "expected": int(actual) if actual.is_integer() else actual,
                    "found": int(claimed) if claimed.is_integer() else claimed,
                    "position": m.start()
                })

    def logic_check(self, reasoning: str):
        return

    def contradiction_check(self, reasoning: str):
        return

    def final_report(self):
        return {
            "valid": len(self.errors) == 0,
            "num_errors": len(self.errors),
            "errors": self.errors
        }