"""
router.py
=========

Strategy Router for MNSR (expanded strategy set).

WHY THIS CHANGE
---------------
The original router only executed two real prompts (REVISE, BACKTRACK)
plus one memory-augmented variant. The improved controller now emits ten
distinct actions, each of which implies a genuinely different repair
*strategy*, not just a re-worded prompt. This module implements each
strategy as its own method with an explicit "selection condition"
docstring, and keeps `execute()` as a pure dispatcher -- consistent with
the original design note "the router never decides, it only executes."

STRATEGIES IMPLEMENTED
-----------------------
* Chain of Thought        -- implicit: the initial model.reasoning() call
                              in pipeline.py; not re-executed here.
* Answer Repair            -- ANSWER_REPAIR: re-derive only the final
                              answer from an already-sound reasoning trace.
* Reasoning Repair (Step Repair) -- REASONING_REPAIR: rewrite only the
                              flawed steps flagged by the validator.
* Replanning               -- REPLAN: discard and restart with an explicit
                              upfront plan.
* Symbolic Correction / Backtrack -- BACKTRACK: discard and restart from
                              scratch after a hard contradiction.
* Decomposition            -- DECOMPOSE: split into sub-questions, solve
                              each, then recombine.
* Memory-Guided Reflection -- MEMORY_REFLECTION: condition on a retrieved
                              failure episode and its correction.
* Self-Critique            -- SELF_CRITIQUE: critique-then-revise in two
                              explicit sub-steps.
* Self-Verification        -- SELF_VERIFY: independently re-derive the
                              answer and reconcile with the original.
* Majority Voting / Multi-Path Reasoning -- MULTI_PATH_REASONING: sample
                              several independent paths and vote.

RESEARCH NOVELTY
-----------------
Giving each meta-cognitive action its own minimal, targeted prompt (rather
than one generic "please fix this" prompt reused everywhere) is what
allows an ablation over *strategy choice* independent of *whether to
correct at all* -- i.e. the paper can show that routing to the "right"
strategy per failure type outperforms always using the same one
(reported as "MNSR vs. MNSR-uniform-revise" in ablation.py).
"""

from typing import Dict, List
from mnsr.cognitive_state import CognitiveState


class StrategyRouter:
    """
    Executes the action selected by the Meta-Cognitive Controller.
    The router never decides, it only executes.
    """

    def __init__(self, model):
        self.model = model

    def execute(
        self,
        action: str,
        question: str,
        state: CognitiveState,
        memory_hint: str = "",
        failure_hint: str = "",
    ) -> Dict:
        dispatch = {
            "CONTINUE": self._passthrough,
            "TERMINATE": self._passthrough,
            "ANSWER_REPAIR": self._answer_repair,
            "REASONING_REPAIR": self._reasoning_repair,
            "BACKTRACK": self._backtrack,
            "REPLAN": self._replan,
            "DECOMPOSE": self._decompose,
            "MEMORY_REFLECTION": self._memory_reflection,
            "SELF_CRITIQUE": self._self_critique,
            "SELF_VERIFY": self._self_verify,
            "MULTI_PATH_REASONING": self._multi_path_reasoning,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Unknown system execution action: {action}")

        if action in ("CONTINUE", "TERMINATE"):
            return handler(state)
        if action == "MEMORY_REFLECTION":
            return handler(question, state, memory_hint, failure_hint)
        return handler(question, state)

    # ------------------------------------------------------------------
    # No-op passthrough for terminal actions
    # ------------------------------------------------------------------
    def _passthrough(self, state: CognitiveState) -> Dict:
        return {"reasoning": state.reasoning, "answer": state.final_answer}

    # ------------------------------------------------------------------
    # Selection condition: validated & consistent reasoning, but the
    # stated final answer disagrees with the derivation. Cheapest repair:
    # only touch the answer extraction, not the whole reasoning trace.
    # ------------------------------------------------------------------
    def _answer_repair(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""The reasoning below correctly solves the question, but the
stated "Final Answer" line does not match the value actually derived in the
reasoning. Do not change the reasoning. Simply restate the final answer that
is consistent with the derivation.

Question:
{question}

Reasoning:
{state.reasoning}

Respond with exactly:
Final Answer: <answer>"""
        return self._run(prompt)

    # ------------------------------------------------------------------
    # Selection condition: hard symbolic (usually arithmetic) errors
    # present without full contradiction. Fix only the flawed steps.
    # ------------------------------------------------------------------
    def _reasoning_repair(self, question: str, state: CognitiveState) -> Dict:
        errors_desc = "\n".join(
            f"- {e.get('type')}: {e.get('expression', e.get('detail', ''))}"
            for e in state.symbolic_errors
        ) or "No specific errors listed."
        prompt = f"""You are reviewing a mathematical/logical problem solver. The
following reasoning contains specific flagged errors. Correct only the
flawed steps; keep the rest of the reasoning intact.

Question:
{question}

Previous reasoning trail:
{state.reasoning}

Flagged errors:
{errors_desc}

Provide the corrected step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(prompt)

    # ------------------------------------------------------------------
    # Selection condition: severe contradiction / systemic risk, first
    # occurrence this trajectory. Discard everything, start fresh.
    # ------------------------------------------------------------------
    def _backtrack(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""The previous reasoning trail contained fatal symbolic
contradictions or arithmetic errors. Discard the failed thinking strategy
entirely and solve the question completely fresh from the beginning.

Question:
{question}

Provide your completely new step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(prompt)

    # ------------------------------------------------------------------
    # Selection condition: same severe risk as BACKTRACK, but a plain
    # backtrack was already tried and failed -- ask for an explicit plan
    # before executing, to break the failure loop.
    # ------------------------------------------------------------------
    def _replan(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""A prior fresh attempt at this question still failed
validation. Before solving, first write a short numbered PLAN describing
the distinct steps needed. Then execute the plan step by step.

Question:
{question}

Reasoning that still failed:
{state.reasoning}

Format:
Plan:
1. ...
2. ...

Execution:
<step-by-step reasoning following the plan>

Final Answer: <answer>"""
        return self._run(prompt)

    # ------------------------------------------------------------------
    # Selection condition: reasoning is too shallow (few lines) relative
    # to a low-confidence result -- likely needs to be broken into
    # sub-questions rather than repaired in place.
    # ------------------------------------------------------------------
    def _decompose(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""The question below may require multiple sub-steps that were
skipped. Break it into 2-4 smaller sub-questions, answer each sub-question in
order, then combine the sub-answers into the final answer.

Question:
{question}

Format:
Sub-question 1: ...
Sub-answer 1: ...
Sub-question 2: ...
Sub-answer 2: ...
...

Final Answer: <answer>"""
        return self._run(prompt)

    # ------------------------------------------------------------------
    # Selection condition: a similar *failed* past episode was retrieved
    # above the memory-similarity threshold. Condition on both the past
    # mistake and its correction to avoid repeating it.
    # ------------------------------------------------------------------
    def _memory_reflection(
        self, question: str, state: CognitiveState, memory_hint: str, failure_hint: str
    ) -> Dict:
        hint_block = memory_hint or "No successful reference episode available."
        failure_block = failure_hint or "No failure precedent available."
        prompt = f"""You have access to reference memory of past similar
problems. One shows a past MISTAKE to avoid; the other shows a past
CORRECT / successful resolution pattern. Use both to avoid repeating the
mistake on the current target problem.

Past mistake to avoid:
{failure_block}

Past successful resolution pattern:
{hint_block}

Target Question:
{question}

Provide your step-by-step reasoning using the reference knowledge.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(prompt)

    # ------------------------------------------------------------------
    # Selection condition: reasoning-consistency below threshold but no
    # hard symbolic error -- a soft signal best handled by asking the
    # model to critique itself before revising, in two explicit phases.
    # ------------------------------------------------------------------
    def _self_critique(self, question: str, state: CognitiveState) -> Dict:
        critique_prompt = f"""Critique the following reasoning for internal
contradictions, unjustified leaps, or unclear logic. List concrete issues
only; do not solve the question yet.

Question:
{question}

Reasoning:
{state.reasoning}

Critique:"""
        critique_result = self.model.generate(critique_prompt)
        critique_text = critique_result.get("text", "")

        revise_prompt = f"""Using the critique below, rewrite the reasoning to
address every issue raised.

Question:
{question}

Original reasoning:
{state.reasoning}

Critique:
{critique_text}

Provide the corrected step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(revise_prompt)

    # ------------------------------------------------------------------
    # Selection condition: confidence is borderline but validation is
    # clean and this is the first correction attempt -- ask the model to
    # independently re-derive and reconcile, cheaper than a full revise.
    # ------------------------------------------------------------------
    def _self_verify(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""Independently re-derive the answer to the question below
without looking at the previous reasoning. Then compare your new derivation
to the previous answer. If they agree, restate that answer. If they
disagree, explain which is correct and use it as the final answer.

Question:
{question}

Previous answer:
{state.final_answer}

Provide your independent step-by-step reasoning, the comparison, and then:
Final Answer: <answer>"""
        return self._run(prompt)

    # ------------------------------------------------------------------
    # Selection condition: confidence remains low even after a prior
    # correction attempt -- single-path repair is not converging, so
    # sample multiple independent paths and take a majority vote.
    # ------------------------------------------------------------------
    def _multi_path_reasoning(self, question: str, state: CognitiveState) -> Dict:
        paths = self.model.generate_multi_path(question, num_paths=3, temperature=0.7)
        answers = [p["answer"] for p in paths if p.get("answer")]

        if not answers:
            return {"reasoning": state.reasoning, "answer": state.final_answer}

        vote_counts: Dict[str, int] = {}
        for a in answers:
            key = a.strip().lower()
            vote_counts[key] = vote_counts.get(key, 0) + 1

        winner = max(vote_counts, key=vote_counts.get)
        winning_path = next((p for p in paths if p["answer"].strip().lower() == winner), paths[0])

        combined_reasoning = "\n\n---\n\n".join(
            f"[Path {i+1}] {p['reasoning']}" for i, p in enumerate(paths)
        )
        combined_reasoning += f"\n\n[Majority Vote Result] {winning_path['answer']}"

        return {"reasoning": combined_reasoning, "answer": winning_path["answer"]}

    # ------------------------------------------------------------------
    # Shared execution helper
    # ------------------------------------------------------------------
    def _run(self, prompt: str) -> Dict:
        result = self.model.generate(prompt)
        text = result.get("text", "")
        return {"reasoning": text, "answer": self.model.extract_answer(text)}
