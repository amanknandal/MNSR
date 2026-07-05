import os
import sys
import json
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.baseline import BaselineCoT
from mnsr.pipeline import MNSRPipeline
from mnsr.metrics import ExperimentMetrics


class MNSRExperiment:

    BOOL_MAP = {
        "true": "yes",
        "yes": "yes",
        "correct": "yes",
        "false": "no",
        "no": "no",
        "incorrect": "no"
    }

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.baseline = BaselineCoT()
        self.mnsr = MNSRPipeline()
        self.metrics = ExperimentMetrics()
        self.results = []

    def load_dataset(self, max_samples=1000):
        print(f"\nLoading dataset from: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        try:
            data = json.loads(content)

            if isinstance(data, list):
                print(f"Found {len(data)} total samples.")
                return data[:max_samples]

            elif isinstance(data, dict):
                for key in ["data", "examples", "questions", "items", "dataset"]:
                    if key in data:
                        print(f"Found key '{key}' containing {len(data[key])} samples.")
                        return data[key][:max_samples]
        except Exception:
            pass

        try:
            data = []
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))

            print(f"Detected JSONL dataset with {len(data)} samples.")
            return data[:max_samples]

        except Exception as e:
            raise RuntimeError(f"Unable to load dataset: {e}")

    def normalize_sample(self, sample):
        question = None

        QUESTION_KEYS = [
            "question",
            "input",
            "query",
            "prompt",
            "instruction",
            "text",
        ]

        for key in QUESTION_KEYS:
            if key in sample and sample[key]:
                question = sample[key]
                break

        if question is None:
            raise ValueError(
                f"Unable to find question field.\nAvailable keys: {list(sample.keys())}"
            )

        context_parts = []

        if "knowledge" in sample and sample["knowledge"]:
            context_parts.append(str(sample["knowledge"]).strip())

        if "facts" in sample and sample["facts"]:
            facts = sample["facts"]
            if isinstance(facts, list):
                context_parts.append(" ".join(str(f).strip() for f in facts))
            else:
                context_parts.append(str(facts).strip())

        if context_parts:
            question = f"Context: {' '.join(context_parts)}\n\nQuestion: {question}"

        answer = None

        ANSWER_KEYS = [
            "answer",
            "right_answer",
            "best_answer",
            "correct_answer",
            "gold_answer",
            "target",
            "output",
            "label",
            "reference",
            "expected_answer",
            "ground_truth",
        ]

        for key in ANSWER_KEYS:
            if key in sample and sample[key] is not None and sample[key] != "":
                answer = sample[key]
                break

        correct_answers = None
        incorrect_answers = None

        if "correct_answers" in sample and sample["correct_answers"]:
            correct_answers = [str(a) for a in sample["correct_answers"]]
            if answer is None:
                answer = correct_answers[0]

        if "incorrect_answers" in sample and sample["incorrect_answers"]:
            incorrect_answers = [str(a) for a in sample["incorrect_answers"]]

        if answer is None and "references" in sample:
            ans = sample["references"]
            answer = ans[0] if isinstance(ans, list) else ans

        if answer is None and "answerKey" in sample:
            answer = sample["answerKey"]

        if answer is None:
            raise ValueError(
                f"Unable to determine answer field.\nAvailable keys: {list(sample.keys())}"
            )

        if isinstance(answer, bool):
            answer = "yes" if answer else "no"

        return {
            "question": str(question),
            "answer": str(answer),
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers
        }

    def normalize_answer(self, answer) -> str:
        if answer is None:
            return ""
        ans_str = str(answer).strip().lower()
        if "####" in ans_str:
            ans_str = ans_str.split("####")[-1].strip()
        elif "final answer:" in ans_str:
            ans_str = ans_str.split("final answer:")[-1].strip()
        ans_str = ans_str.strip(" .\"'")
        return self.BOOL_MAP.get(ans_str, ans_str)

    def is_correct(self, prediction, gold, correct_answers=None, incorrect_answers=None) -> bool:
        pred_norm = self.normalize_answer(prediction)
        gold_norm = self.normalize_answer(gold)

        if re.fullmatch(r'-?\d+\.?\d*', gold_norm):
            pred_numbers = re.findall(r'-?\d+\.?\d*', pred_norm)
            if not pred_numbers:
                return False
            try:
                return float(pred_numbers[-1]) == float(gold_norm)
            except ValueError:
                return False

        if gold_norm in ("yes", "no"):
            pred_bool = None
            for word in pred_norm.split():
                if word in self.BOOL_MAP:
                    pred_bool = self.BOOL_MAP[word]
                    break
            return pred_bool == gold_norm

        if correct_answers:
            norm_corrects = [self.normalize_answer(a) for a in correct_answers]
            hit_correct = any(c and (c in pred_norm or pred_norm in c) for c in norm_corrects)
            if incorrect_answers:
                norm_incorrects = [self.normalize_answer(a) for a in incorrect_answers]
                hit_incorrect = any(c and (c in pred_norm or pred_norm in c) for c in norm_incorrects)
                if hit_incorrect and not hit_correct:
                    return False
            return hit_correct

        if not gold_norm:
            return False
        return gold_norm in pred_norm or pred_norm in gold_norm

    def process_single_trajectory(self, args):
        idx, sample, total_count = args

        sample = self.normalize_sample(sample)

        question = sample["question"]
        gold = sample["answer"]
        correct_answers = sample.get("correct_answers")
        incorrect_answers = sample.get("incorrect_answers")

        t0 = time.time()
        baseline_result = self.baseline.solve(question)
        baseline_time = time.time() - t0

        t1 = time.time()
        mnsr_result = self.mnsr.solve(question)
        mnsr_time = time.time() - t1

        baseline_answer = baseline_result.get("answer", "")
        mnsr_answer = mnsr_result.get("answer", "")

        baseline_correct = self.is_correct(baseline_answer, gold, correct_answers, incorrect_answers)
        mnsr_correct = self.is_correct(mnsr_answer, gold, correct_answers, incorrect_answers)

        print(
            f"[{idx+1}/{total_count}] "
            f"Baseline={baseline_correct} | "
            f"MNSR={mnsr_correct}"
        )

        return {
            "baseline_correct": baseline_correct,
            "mnsr_correct": mnsr_correct,
            "validation_report": mnsr_result.get("validation", {}),
            "final_action": mnsr_result.get("final_action", "none"),
            "steps": mnsr_result.get("state", {}).get("current_step", 1),
            "log_entry": {
                "question": question,
                "gold": gold,
                "baseline_answer": baseline_answer,
                "baseline_correct": baseline_correct,
                "baseline_latency_sec": round(baseline_time, 3),
                "mnsr_answer": mnsr_answer,
                "mnsr_correct": mnsr_correct,
                "mnsr_latency_sec": round(mnsr_time, 3),
                "action_path": mnsr_result.get("final_action", "none"),
                "validation": mnsr_result.get("validation", {})
            }
        }

    def run(self):
        dataset = self.load_dataset(max_samples=1000)
        total_count = len(dataset)
        print(f"\nLoaded {total_count} evaluation samples. Initializing concurrent worker tasks...\n")

        task_queue = [(idx, sample, total_count) for idx, sample in enumerate(dataset)]

        MAX_WORKERS = 4

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            compiled_outputs = list(executor.map(self.process_single_trajectory, task_queue))

        print("\nAggregating experimental results and compiling metrics...")
        for out in compiled_outputs:
            self.metrics.update(
                baseline_correct=out["baseline_correct"],
                mnsr_correct=out["mnsr_correct"],
                validation_report=out["validation_report"],
                final_action=out["final_action"],
                steps=out["steps"]
            )
            self.results.append(out["log_entry"])

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