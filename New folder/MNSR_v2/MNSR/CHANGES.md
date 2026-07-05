# MNSR v2 — Summary of Modifications

This revision preserves the original architecture:

```
Question → Initial Reasoning → Symbolic Validator → Memory Retrieval
→ Meta-Cognitive Controller → Strategy Router → Iterative Self-Correction
→ Final Validation → Memory Update
```

Every module was extended in place; no module was removed or replaced with
a different paradigm. Each file carries a module-level docstring
explaining *why* the change was made, its *research novelty*, and its
expected *benchmark impact* — summarized here for convenience.

| # | Module | Change | Why it matters |
|---|--------|--------|-----------------|
| 1 | `mnsr/confidence.py` **(new)** | Multi-signal `ConfidenceEstimator` replaces the hardcoded `0.80` | Confidence now causally reflects validation score, consistency, error count, memory match, retries, optional LLM self-eval — a prerequisite for any real meta-cognition claim |
| 2 | `mnsr/symbolic_validator.py` | 9 checks (arithmetic, contradiction, unsupported assumptions, hallucination heuristic, missing steps, answer format, final-answer consistency, dataset constraints) + structured `{valid, score, errors, warnings, reasoning_consistency, answer_consistency}` report | Catches non-arithmetic failure modes dominant in StrategyQA/TruthfulQA/HaluEval, not just GSM8K arithmetic slips |
| 3 | `mnsr/controller.py` | 11-action deterministic priority-ordered decision function (TERMINATE, BACKTRACK, REPLAN, ANSWER_REPAIR, REASONING_REPAIR, DECOMPOSE, MEMORY_REFLECTION, SELF_VERIFY, SELF_CRITIQUE, MULTI_PATH_REASONING, CONTINUE) using confidence + validation score + consistency + memory + correction history + retry count | Lets the paper report *which* strategy fires for *which* failure signature instead of a single generic "revise" |
| 4 | `mnsr/memory.py` | Stores both success and failure episodes; retrieval returns best match of each type separately | Enables genuine error-avoidance (recognizing "I was wrong on something like this before") rather than only post-hoc correction |
| 5 | `mnsr/pipeline.py` | Bounded iterative reflection loop with a monotone "best-so-far" guard, per-step trace log, dataset-type inference, dual success/failure memory threading | Prevents the known failure mode where naive self-refinement *degrades* an already-correct answer; adds full auditability |
| 6 | `mnsr/router.py` | 10 distinct strategy implementations (Answer Repair, Reasoning Repair, Backtrack, Replan, Decompose, Memory-Guided Reflection, Self-Critique, Self-Verify, Multi-Path/Majority-Vote) each with an explicit selection-condition docstring | Matches action space to controller; enables the "targeted strategy vs. uniform revise" ablation |
| 7 | `mnsr/cognitive_state.py` | Expanded state tuple: validation score, reasoning/answer consistency, memory match type, correction history (tabu list), retry count, elapsed time | Supports every new controller/router signal without breaking the original `to_dict()`/`reset()` contract |
| 8 | `mnsr/metrics.py` | Generic per-action counters (was: 3 hardcoded), reflection depth, avg confidence, validation success rate, memory retrieval rate, correction success rate | Matches the full "Evaluation Improvements" metric list |
| 9 | `evaluation/experiment.py` | Filename-based dataset-type inference (numeric/boolean/multiple_choice/freeform) threaded into both baseline and MNSR solves; MMLU/ARC choices folded into the prompt; richer metrics update call | Adds MMLU/ARC/BoolQ support and dataset-aware validation without hardcoding per-dataset branches in the pipeline itself |
| 10 | `evaluation/baseline.py` | Baselines accept the same `dataset_hint`, use `MNSRPipeline.infer_dataset_type` for validation | Keeps baseline-vs-MNSR comparisons apples-to-apples |
| 11 | `evaluation/benchmark.py` | Generic per-action-type breakdown, exact-match rate, average confidence, correction rate | Matches expanded action space; adds Exact Match + Avg Confidence per spec |
| 12 | `evaluation/ablation.py` | Fixed `MockValidator` signature (`dataset_type` kwarg); added `MockConfidenceEstimator` for a new "- Dynamic Confidence" ablation arm | Old mocks would now crash against the new pipeline; new ablation directly isolates the improvement most emphasized in the spec |
| 13 | `evaluation/visualize.py` | Accepts `output_dir` so each dataset run's figures land in `results/<dataset>/` | Matches `run.py`'s existing call signature (previously would raise `TypeError`) |
| 14 | `models/phi3.py` | Added `temperature` override, `self_evaluate()`, `generate_multi_path()` | Backing capabilities for SELF_VERIFY / MULTI_PATH_REASONING and the optional LLM self-eval confidence signal |

## Backward compatibility

- All public method signatures are additive (new parameters have
  defaults); existing callers (e.g. `run.py`) work unmodified.
- `ReflectionMemory.retrieve_best()` preserves the original
  single-best-match API for any external code still depending on it.
- `CognitiveState.to_dict()` retains every original key.

## Known simplifications (flagged for transparency)

- Contradiction/hallucination/assumption checks are regex-cue heuristics,
  not an NLI model — cheap and dependency-free, but will miss
  paraphrased contradictions. A natural follow-up is swapping in a small
  NLI classifier behind the same `SymbolicValidator` interface.
- `ReflectionMemory` still uses TF-IDF vectors rather than dense
  embeddings; swapping in a sentence-embedding model is a drop-in change
  behind `_rebuild_index`/`retrieve` without touching any caller.
