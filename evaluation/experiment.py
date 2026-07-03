import os
import sys
import json
import time
import re
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.baseline import BaselineCoT
from mnsr.pipeline import MNSRPipeline
from mnsr.metrics import ExperimentMetrics
class MNSRExperiment:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.baseline = BaselineCoT()
        self.mnsr = MNSRPipeline()
        self.metrics = ExperimentMetrics()
        self.results = []
    def load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    def normalize_answer(self, answer) -> str:
        if answer is None:
            return ""
        ans_str = str(answer).strip().lower()
        if "final answer:" in ans_str:
            ans_str = ans_str.split("final answer:")[-1].strip()
        match = re.search(r'(-?\d+\.?\d*)', ans_str)
        if match:
            return match.group(1)
        return ans_str
    def is_correct(self, prediction, gold) -> bool:
        return self.normalize_answer(prediction) == self.normalize_answer(gold)
    def run(self):
        dataset = self.load_dataset()
        print(f"\nLoaded {len(dataset)} evaluation samples.\n")
        for idx, sample in enumerate(dataset):
            question = sample["question"]
            gold = sample["answer"]
            print(f"[{idx+1}/{len(dataset)}] Processing current evaluation trajectory...")
            t0 = time.time()
            baseline_result = self.baseline.solve(question)
            baseline_time = time.time() - t0
            t1 = time.time()
            mnsr_result = self.mnsr.solve(question)
            mnsr_time = time.time() - t1
            baseline_correct = self.is_correct(baseline_result.get("answer"), gold)
            mnsr_correct = self.is_correct(mnsr_result.get("answer"), gold)
            self.metrics.update(
                baseline_correct=baseline_correct,
                mnsr_correct=mnsr_correct,
                validation_report=mnsr_result["validation"],
                final_action=mnsr_result["final_action"],
                steps=mnsr_result["state"]["current_step"]
            )
            self.results.append({
                "question": question,
                "gold": gold,
                "baseline_answer": baseline_result.get("answer"),
                "baseline_correct": baseline_correct,
                "baseline_latency_sec": round(baseline_time, 3),
                "mnsr_answer": mnsr_result.get("answer"),
                "mnsr_correct": mnsr_correct,
                "mnsr_latency_sec": round(mnsr_time, 3),
                "action_path": mnsr_result["final_action"],
                "validation": mnsr_result["validation"]
            })
        return self.results
    def save_results(self, filename="results.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults logs safely exported to {filename}")
    def print_summary(self):
        print("\n" + "="*10 + " MNSR EXPERIMENT RUN SUMMARY " + "="*10 + "\n")
        summary = self.metrics.summary()
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k:23}: {sub_v}")
            else:
                print(f"{k:25}: {v}")
        print("\n" + "="*49 + "\n")