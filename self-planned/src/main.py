import asyncio
from typing import Any, Optional
from dotenv import load_dotenv
import pandas as pd
import random
from planner import create_planner, is_schema_too_generic, refine_schema
from executor import run_plan

load_dotenv()


def fetch_sample(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} samples")

    # Get a random sample
    sample_idx = random.randint(0, len(df) - 1)
    sample = df.iloc[sample_idx]

    print("\n=== SAMPLE FROM TEST DATASET ===")
    print(f"Index: {sample_idx}")
    print(f"Input: {sample['input']}")
    print(f"Label: {sample['label']}")
    print(f"Num Variables: {sample['num_variables']}")
    print(f"Template: {sample['template']}")
    print("=" * 50)

    return sample
   

async def run_planner(sample: pd.Series) -> Optional[dict[str, Any]]:
    """Run the planner on a sample and return the generated plan."""

    print("\n=== RUNNING PLANNER ===")

    # Prepare task description first
    task_description = """
Task: Given a natural-language input that contains a Premise and a Hypothesis, decide whether the Hypothesis is True or False under the Peter-Clark (PC) algorithm.

- PC is a constraint-based causal discovery method that infers a causal equivalence class (CPDAG) from observational (in)dependence information.
- Before deciding, reconstruct a global causal structure over all variables mentioned in the Premise; do NOT rely on pairwise or local checks.
- Return True only if the claim holds in every DAG in the Markov equivalence class implied by the Premise; otherwise return False.

Your plan must mirror the canonical PC algorithm. Reconstruct a global causal structure over all variables before deciding. Do not base the decision on a single pair or local cues. If your plan deviates from PC semantics, it is invalid.

Input available in context: 'input' (contains premise with variables, correlations, conditional independencies, and hypothesis).

CRITICAL OUTPUT FORMAT: The final stage must output ONLY a boolean value (true or false).
"""

    # Create algorithm-aware planner
    planner = await create_planner(task_description)


    print("📝 TASK DESCRIPTION SENT TO PLANNER:")
    print("-" * 80)
    print(task_description)
    print("-" * 80)

    print("\n⏳ Generating plan...")
    result = await planner.run(task_description)
    plan = result.output

    print(f"\n✅ Plan generated successfully!")
    print(f"📊 Number of stages: {len(plan.stages)}")
    print(f"🔑 Final key: {plan.final_key}")

    # Print the complete generated plan
    # print(f"\n📋 === GENERATED PLAN ===")
    # print(json.dumps(plan.model_dump(), indent=2))

    # Validate and refine schemas
    print("\n🔍 Schema refinement:", end=" ")
    refined_count = 0

    for stage in plan.stages:
        if is_schema_too_generic(stage.output_schema):
            try:
                stage.output_schema = await refine_schema(stage.id, stage.writes, stage.prompt_template)
                refined_count += 1
                print("✅", end="")
            except Exception:
                print("❌", end="")
        else:
            print("✓", end="")

    print(f" ({refined_count}/{len(plan.stages)} refined)")

    print(f"\n📋 Plan: {len(plan.stages)} stages → {plan.final_key}")

    # Show context flow first
    print("\n🌊 Context Flow:")
    all_keys = {"input"}
    print(f"  Start: {sorted(all_keys)}")

    for i, stage in enumerate(plan.stages, 1):
        missing = set(stage.reads) - all_keys
        if missing:
            print(f"  ⚠️  Stage {i} missing: {missing}")
        all_keys.update(stage.writes)
        print(f"  Stage {i}: {sorted(all_keys)}")

    # Show detailed stage breakdown
    print(f"\n📝 Stage Details:")
    for i, stage in enumerate(plan.stages, 1):
        print(f"  {i}. {stage.id}")
        print(f"     Reads: {stage.reads}")
        print(f"     Writes: {stage.writes}")
        print(f"     Prompt:")
        # Show full prompt with proper indentation
        for line in stage.prompt_template.split('\n'):
            print(f"       {line}")
        print()

    return plan.model_dump()


async def run_complete_workflow(sample: pd.Series) -> Optional[dict[str, Any]]:
    """Run the complete workflow: planner + executor."""

    # Step 1: Generate plan
    plan_dict = await run_planner(sample)
    if plan_dict is None:
        return None

    # Convert back to Plan object (since run_planner returns dict)
    from models import Plan
    plan = Plan.model_validate(plan_dict)

    # Step 2: Execute plan
    print("\n" + "="*50)
    initial_context = {"input": sample['input']}
    final_context = await run_plan(plan, initial_context)

    # Step 3: Extract final result
    final_key = plan.final_key or "hypothesis_result"
    final_result = final_context.get(final_key)

    print(f"\n🎯 FINAL RESULT: {final_result}")
    print(f"Expected: {bool(sample['label'])}")

    return {
        "plan": plan_dict,
        "final_context": final_context,
        "final_result": final_result,
        "expected": sample.get('label', 'Unknown')
    }


async def main() -> None:
    csv_path = "../data/test_dataset.csv"
    sample = fetch_sample(csv_path)
    await run_complete_workflow(sample)


if __name__ == "__main__":
    asyncio.run(main())
