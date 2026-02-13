import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from tasks.base import BaseTask


class CausalDiscoveryTask(BaseTask):
    """Task configuration for causal discovery using the Peter-Clark (PC) algorithm."""

    algorithm_name = "Peter-Clark (PC) Algorithm"

    task_description = """
Task: Given a natural-language input that contains a Premise and a Hypothesis, decide whether the Hypothesis is True or False under the Peter-Clark (PC) algorithm.

- PC is a constraint-based causal discovery method that infers a causal equivalence class (CPDAG) from observational (in)dependence information.
- Before deciding, reconstruct a global causal structure over all variables mentioned in the Premise; do NOT rely on pairwise or local checks.
- Return True only if the claim holds in every DAG in the Markov equivalence class implied by the Premise; otherwise return False.

ENVIRONMENT (VERY IMPORTANT):
- You do NOT have a dataset and you MUST NOT propose to run new statistical CI tests.
- All (in)dependence information is given EXPLICITLY in the Premise as text. Treat this as a PERFECT CI oracle.
- The Premise will contain sentences like:
    • "X correlates with Y"       → treat as: X and Y are dependent; there is an adjacency between X and Y.
    • "X is independent of Y"    → treat as: X ⟂ Y | ∅.
    • "X and Y are independent given Z" or
      "X and Y are independent given Z and W and ..."
                                   → treat as: X ⟂ Y | {Z, W, ...}.
- The Premise claims to list ALL relevant statistical relations among the variables. You must therefore:
    • Trust that if an independence X ⟂ Y | S is stated, it is true.
    • NOT invent independencies that are not mentioned.
    • When the PC algorithm conceptually "calls" CI(X, Y | S), answer it by checking whether the Premise explicitly states
      that X and Y are independent given exactly S (or ∅); otherwise treat them as dependent under that conditioning set.
- Do NOT generate or enumerate arbitrary conditioning sets beyond those explicitly mentioned in the Premise. You may only rely on
  the conditioning sets that appear in the text.

ALGORITHM REQUIREMENT:
- Your plan must mirror the canonical Peter-Clark (PC) algorithm, and uses of CI(i, j | S) must be implemented via LOOKUP into the Premise as described above, not via new tests.
- The decision MUST be based on the global causal structure (CPDAG) over all variables, not on a single pair or local cues.

Input available in context: "input" (contains premise with variables, correlations, conditional independencies, and hypothesis).

CONSERVATIVE DECISION-MAKING (VERY IMPORTANT):
- DEFAULT TO FALSE: Return True ONLY if the hypothesis is DEFINITIVELY and UNAMBIGUOUSLY supported by the reconstructed structure.
- REQUIRE EXPLICIT EVIDENCE: You must be able to trace a clear path from the input data → through each algorithmic step → to the conclusion.
- HANDLE UNCERTAINTY CONSERVATIVELY: If ANY step produces ambiguous or uncertain results (e.g., edge orientation is undetermined), the final answer should be False.
- VERIFY ALL CONDITIONS: The hypothesis must satisfy ALL its conditions, not just some. For example:
    • "A directly causes B" requires a DEFINITE directed edge A→B (not A-B undirected, not A←B)
    • "A indirectly causes B" requires a DEFINITE directed path A→...→B with NO direct edge
    • "X is a confounder of A and B" requires DEFINITE edges X→A and X→B
- WHEN IN DOUBT, RETURN FALSE: If you cannot definitively confirm the hypothesis from the constructed structure, return False.
- EQUIVALENCE CLASS AWARENESS: Remember that undirected edges represent uncertainty - the true direction could go either way. Only directed edges provide definitive evidence.

CRITICAL OUTPUT FORMAT:
- The final stage MUST output a SIMPLE BOOLEAN VALUE (true or false), NOT a nested object.
- The output should be a single key with a boolean value, for example: {"decision": true} or {"result": false}
- DO NOT output nested objects like {"decision": {"holds": true, "explanation": "..."}}
- DO NOT include explanation fields - just the boolean decision.
- The schema for the final stage MUST specify a simple boolean type, not an object type.
"""

    default_dataset_path = "../../data/test_dataset.csv"

    def load_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        print(f"Dataset loaded: {len(df)} samples")

        required_cols = ["input", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def fetch_sample(self, dataset: pd.DataFrame, sample_idx: Optional[int] = None) -> pd.Series:
        if sample_idx is not None:
            if sample_idx < 0 or sample_idx >= len(dataset):
                raise ValueError(f"Sample index {sample_idx} out of range (0-{len(dataset)-1})")
            actual_idx = sample_idx
            print(f"Using specified index: {actual_idx}")
        else:
            actual_idx = random.randint(0, len(dataset) - 1)
            print(f"Using random index: {actual_idx}")

        sample = dataset.iloc[actual_idx]

        print(f"Index: {actual_idx}")
        print(f"Input: {sample['input']}")
        print(f"Label: {sample['label']}")
        print(f"Num Variables: {sample['num_variables']}")
        print(f"Template: {sample['template']}")
        print("=" * 50)

        return sample

    def extract_result(
        self,
        final_result: Any,
        final_key: str,
        plan: Any,
    ) -> Dict[str, Any]:
        stage_id = plan.stages[-1].id if plan.stages else "unknown"
        actual, warning = _extract_boolean_from_result(final_result, final_key, stage_id)

        if warning:
            print(warning)

        return {"predicted": actual}

    def evaluate(
        self,
        extracted: Dict[str, Any],
        sample: pd.Series,
    ) -> Dict[str, Any]:
        expected = bool(sample["label"])
        predicted = extracted["predicted"]
        is_correct = predicted == expected

        return {
            "is_correct": is_correct,
            "predicted_summary": str(predicted),
            "expected_summary": str(expected),
            "predicted": predicted,
            "expected": expected,
        }

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [r for r in results if r.get("error") is None]
        if not successful:
            return {"accuracy": 0.0, "total": 0}

        total = len(successful)
        correct = sum(1 for r in successful if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        true_positives = sum(1 for r in successful if r.get("expected") and r.get("predicted"))
        false_positives = sum(1 for r in successful if not r.get("expected") and r.get("predicted"))
        false_negatives = sum(1 for r in successful if r.get("expected") and not r.get("predicted"))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        print(f"Accuracy:              {metrics['accuracy']*100:.1f}%")
        print(f"Precision:             {metrics.get('precision', 0)*100:.1f}%")
        print(f"Recall:                {metrics.get('recall', 0)*100:.1f}%")
        print(f"F1 Score:              {metrics.get('f1_score', 0)*100:.1f}%")


def _extract_boolean_from_result(
    final_result: Any,
    final_key: str,
    stage_id: str = "final_stage",
) -> Tuple[bool, Optional[str]]:
    """Extract boolean from various formats with algorithm-agnostic heuristics."""
    warning = None

    if isinstance(final_result, bool):
        return final_result, None

    if isinstance(final_result, str):
        lower = final_result.lower().strip()
        if lower in ["true", "1", "yes"]:
            return True, None
        elif lower in ["false", "0", "no"]:
            return False, None
        raise ValueError(f"Cannot extract boolean from string '{final_result}'")

    if isinstance(final_result, int):
        if final_result in [0, 1]:
            return bool(final_result), None
        raise ValueError(f"Cannot extract boolean from int {final_result}. Expected 0 or 1")

    if isinstance(final_result, dict):
        warning = (
            f"WARNING: Final stage '{stage_id}' returned nested object. "
            f"Attempting intelligent extraction..."
        )

        candidates = [
            "verified", "holds", "decision", "result", "answer",
            "is_true", "value", "valid", "correct", "output",
        ]

        for field in candidates:
            if field in final_result and isinstance(final_result[field], bool):
                return final_result[field], warning

        for key, value in final_result.items():
            if isinstance(value, bool):
                warning += f"\nUsing first boolean field: '{key}' = {value}"
                return value, warning

        for key, value in final_result.items():
            if isinstance(value, dict):
                try:
                    nested_bool, _ = _extract_boolean_from_result(value, key, f"{stage_id}.{key}")
                    warning += f"\nUsing nested boolean from '{key}'"
                    return nested_bool, warning
                except ValueError:
                    continue

        raise ValueError(
            f"Cannot extract boolean from object: {final_result}. "
            f"No boolean fields found."
        )

    if isinstance(final_result, list):
        if len(final_result) == 1:
            return _extract_boolean_from_result(final_result[0], final_key, stage_id)
        raise ValueError(f"Cannot extract boolean from array: {final_result}")

    if final_result is None:
        raise ValueError("Cannot extract boolean from None")

    warning = "CRITICAL: Using Python truthiness. Unreliable!"
    return bool(final_result), warning
