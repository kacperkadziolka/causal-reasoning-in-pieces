import asyncio
import argparse
from typing import Any, Optional, Dict, Tuple
from dotenv import load_dotenv
import pandas as pd
import random

from knowledge.extractor import EnhancedKnowledgeExtractor
from plan.iterative_planner import IterativePlanner
from plan.multi_agent_planner import MultiAgentPlanner
from plan.models import Plan
from execute.executor import run_plan
from validate.validator_generator import GenericValidatorGenerator
from utils.logging_config import init_logger

load_dotenv()


TASK_ALGORITHM = "Peter-Clark (PC) Algorithm"
TASK_DESCRIPTION = """
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


def extract_boolean_from_result(
    final_result: Any,
    final_key: str,
    stage_id: str = "final_stage"
) -> Tuple[bool, Optional[str]]:
    """
    Extract boolean from various formats with algorithm-agnostic heuristics.

    Handles:
    - Simple booleans: True/False
    - Strings: "true"/"false", "1"/"0", "yes"/"no"
    - Integers: 1/0
    - Nested objects: {"verified": bool}, {"holds": bool}
    - Arrays: [bool]

    Returns:
        (extracted_boolean, warning_message)
        warning_message is None for clean extractions

    Raises:
        ValueError: If boolean cannot be extracted
    """
    warning = None

    # Case 1: Already boolean (expected after primary fix)
    if isinstance(final_result, bool):
        return final_result, None

    # Case 2: String
    if isinstance(final_result, str):
        lower = final_result.lower().strip()
        if lower in ['true', '1', 'yes']:
            return True, None
        elif lower in ['false', '0', 'no']:
            return False, None
        raise ValueError(f"Cannot extract boolean from string '{final_result}'")

    # Case 3: Integer
    if isinstance(final_result, int):
        if final_result in [0, 1]:
            return bool(final_result), None
        raise ValueError(f"Cannot extract boolean from int {final_result}. Expected 0 or 1")

    # Case 4: Nested object (SHOULD NOT HAPPEN after primary fix)
    if isinstance(final_result, dict):
        warning = (
            f"⚠️  WARNING: Final stage '{stage_id}' returned nested object. "
            f"Attempting intelligent extraction..."
        )

        # Strategy 1: Common boolean field names (algorithm-agnostic)
        candidates = [
            'verified', 'holds', 'decision', 'result', 'answer',
            'is_true', 'value', 'valid', 'correct', 'output'
        ]

        for field in candidates:
            if field in final_result and isinstance(final_result[field], bool):
                return final_result[field], warning

        # Strategy 2: Find ANY boolean field (truly algorithm-agnostic)
        for key, value in final_result.items():
            if isinstance(value, bool):
                warning += f"\n⚠️  Using first boolean field: '{key}' = {value}"
                return value, warning

        # Strategy 3: Nested search
        for key, value in final_result.items():
            if isinstance(value, dict):
                try:
                    nested_bool, _ = extract_boolean_from_result(value, key, f"{stage_id}.{key}")
                    warning += f"\n⚠️  Using nested boolean from '{key}'"
                    return nested_bool, warning
                except ValueError:
                    continue

        raise ValueError(
            f"Cannot extract boolean from object: {final_result}. "
            f"No boolean fields found."
        )

    # Case 5: Array
    if isinstance(final_result, list):
        if len(final_result) == 1:
            return extract_boolean_from_result(final_result[0], final_key, stage_id)
        raise ValueError(f"Cannot extract boolean from array: {final_result}")

    # Case 6: None
    if final_result is None:
        raise ValueError("Cannot extract boolean from None")

    # Fallback
    warning = "⚠️  CRITICAL: Using Python truthiness. Unreliable!"
    return bool(final_result), warning


def fetch_sample(csv_path: str, sample_idx: Optional[int] = None) -> pd.Series:
    """Fetch a sample from the dataset by index or random if not specified"""
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} samples")

    # Get specified sample or random sample
    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= len(df):
            raise ValueError(f"Sample index {sample_idx} out of range (0-{len(df)-1})")
        actual_idx = sample_idx
        print(f"Using specified index: {actual_idx}")
    else:
        actual_idx = random.randint(0, len(df) - 1)
        print(f"Using random index: {actual_idx}")

    sample = df.iloc[actual_idx]

    print(f"Index: {actual_idx}")
    print(f"Input: {sample['input']}")
    print(f"Label: {sample['label']}")
    print(f"Num Variables: {sample['num_variables']}")
    print(f"Template: {sample['template']}")
    print("=" * 50)

    return sample


async def run_simple_workflow(
    sample: pd.Series,
    use_sequential_generation: bool = False,
    use_multi_agent_planner: bool = False,
    cached_plan: Optional[Plan] = None,
    cached_knowledge: Optional[str] = None,
    cached_structured_constraints: Optional[Dict] = None,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run the main workflow: knowledge extraction → planning → execution.

    Args:
        sample: A pandas Series containing the test sample data
        use_sequential_generation: If True, generate stage prompts sequentially
                                   (only used with multi-agent planner)
        use_multi_agent_planner: If True, use MultiAgentPlanner instead of IterativePlanner
        cached_plan: Optional pre-generated plan to skip planning phase
        cached_knowledge: Optional pre-extracted knowledge to skip extraction phase
        cached_structured_constraints: Optional pre-extracted constraints for validation
        verbose: If True, show detailed execution logs; if False, show minimal output

    Returns:
        Dictionary with execution results or None if failed
    """

    if verbose:
        print("\n🚀 WORKFLOW")
        print("=" * 60)

    task_description = TASK_DESCRIPTION

    # STEP 1: Knowledge Extraction (skip if cached)
    if cached_knowledge is not None:
        if verbose:
            print("📦 Using cached knowledge")
        knowledge = cached_knowledge
        structured_constraints = cached_structured_constraints
    else:
        if verbose:
            print("\n📚 STEP 1: Knowledge Extraction")
            print("-" * 30)
        extractor = EnhancedKnowledgeExtractor()
        knowledge, structured_constraints = await extractor.extract_simple_knowledge(TASK_ALGORITHM, sample["input"])
        if verbose:
            print(f"✅ Knowledge extracted: {knowledge}")

    # STEP 2: Planning (skip if cached)
    if cached_plan is not None:
        if verbose:
            print("📦 Using cached plan")
        plan = cached_plan
    else:
        if verbose:
            print("\n🔄 STEP 2: Planning")
            print("-" * 30)

        if use_multi_agent_planner:
            if verbose:
                print("Using MultiAgentPlanner" + (" (SEQUENTIAL mode)" if use_sequential_generation else " (BATCH mode)"))
            planner = MultiAgentPlanner()
            plan, metadata = await planner.generate_plan(
                task_description=task_description,
                algorithm_knowledge=knowledge,
                use_sequential=use_sequential_generation
            )
            if verbose:
                print(f"✅ Planning successful: {len(plan.stages)} stages")
                print(f"   Generation mode: {metadata.get('generation_mode', 'unknown')}")
        else:
            if verbose:
                print("Using IterativePlanner (two-stage planning)")
            planner = IterativePlanner()
            plan = await planner.generate_two_stage_plan(
                task_description=task_description,
                algorithm_knowledge=knowledge,
                enhance_prompts=True
            )
            if verbose:
                print(f"✅ Planning successful: {len(plan.stages)} stages")

    if verbose:
        print("\n⚡ STEP 3: Execution")
        print("-" * 30)

    # Create validator from structured constraints (if available)
    validator = None
    if structured_constraints and structured_constraints.get("stage_constraints"):
        validator = GenericValidatorGenerator(structured_constraints)
        if verbose:
            print(f"   🔒 Validator created with {len(structured_constraints.get('stage_constraints', {}))} stage constraints")

    initial_context = {"input": sample["input"]}
    final_context = await run_plan(plan, initial_context, verbose=verbose, validator=validator)
    final_key = plan.final_key or "result"
    final_result = final_context.get(final_key)

    if verbose:
        print("✅ Execution completed")
        print(f"🎯 Final result key: '{final_key}'")
        print(f"📊 Final result: {final_result}")

        print("\n📊 STEP 4: Evaluation")
        print("-" * 30)
    expected = bool(sample['label'])

    # Convert final_result to boolean using intelligent extraction
    try:
        actual, extraction_warning = extract_boolean_from_result(
            final_result,
            final_key,
            stage_id=plan.stages[-1].id if plan.stages else "unknown"
        )

        if extraction_warning:
            print(extraction_warning)

    except ValueError as e:
        print(f"❌ Boolean extraction failed: {e}")
        print(f"🔍 Final result: {final_result}")
        raise

    is_correct = actual == expected

    if verbose:
        print(f"🎯 Predicted: {actual}")
        print(f"📊 Expected: {expected}")
        print(f"✅ Result: {'CORRECT' if is_correct else 'INCORRECT'}")

    return {
        "sample_idx": sample.name,
        "knowledge_length": len(knowledge),
        "num_stages": len(plan.stages),
        "predicted": actual,
        "expected": expected,
        "is_correct": is_correct,
        "final_context": final_context,
    }


async def main(
    sample_idx: Optional[int] = None,
    use_sequential_generation: bool = False,
    use_multi_agent_planner: bool = False
):
    """
    Main entry point for running the pipeline on a single sample.

    Args:
        sample_idx: Specific sample index or None for random
        use_sequential_generation: Enable sequential stage generation
        use_multi_agent_planner: Use MultiAgentPlanner instead of IterativePlanner
    """
    print("🔬 SELF-PLANNED PIPELINE")
    print("=" * 60)

    if use_multi_agent_planner:
        print(f"🧠 Planner: MultiAgentPlanner ({'SEQUENTIAL' if use_sequential_generation else 'BATCH'} mode)")
    else:
        print("🧠 Planner: IterativePlanner (two-stage)")
    print("=" * 60)

    # Load sample
    csv_path = "../data/test_dataset.csv"
    sample = fetch_sample(csv_path, sample_idx)

    # Run workflow
    result = await run_simple_workflow(
        sample,
        use_sequential_generation=use_sequential_generation,
        use_multi_agent_planner=use_multi_agent_planner
    )

    # Final summary
    print("\n" + "=" * 60)
    print("📋 PIPELINE SUMMARY")
    print("=" * 60)

    if result:
        print("✅ Workflow completed successfully!")
        print(f"📚 Knowledge: {result['knowledge_length']} chars")
        print(f"📝 Stages: {result['num_stages']}")
        print(f"🎯 Prediction: {result['predicted']} (expected: {result['expected']})")
        print(f"📊 Accuracy: {'✅ CORRECT' if result['is_correct'] else '❌ INCORRECT'}")
    else:
        print("❌ Workflow failed")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run self-planned pipeline on specific dataset samples")
    parser.add_argument("--sample-idx", type=int, default=110, help="Specific sample index to test")
    parser.add_argument(
        "--sequential-generation",
        action="store_true",
        default=True,
        help="Generate stage prompts sequentially (only with --multi-agent-planner)"
    )
    parser.add_argument(
        "--multi-agent-planner",
        action="store_true",
        default=True,
        help="Use MultiAgentPlanner instead of IterativePlanner"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode with verbose logging"
    )

    args = parser.parse_args()

    # Initialize logger with debug setting
    init_logger(debug=args.debug)

    if args.sample_idx is not None:
        print(f"🎯 Testing sample index: {args.sample_idx}")
    else:
        print("🎯 Using random sample")

    if args.debug:
        print("🐛 Debug mode enabled - showing verbose logs")

    asyncio.run(main(
        args.sample_idx,
        use_sequential_generation=args.sequential_generation,
        use_multi_agent_planner=args.multi_agent_planner
    ))
