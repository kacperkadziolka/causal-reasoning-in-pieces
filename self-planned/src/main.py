import asyncio
import json
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

    # Create planner
    planner = create_planner()

    # Prepare task description
    task_description = """
Task: Apply the PC algorithm to determine if the hypothesis is True or False.

The PC algorithm follows these steps:
1. Skeleton Discovery: Extract variables and build initial graph from correlations
2. Edge Removal: Remove edges based on conditional independencies
3. V-structure Orientation: Orient v-structures (colliders) using conditional independence
4. Meek Rules: Apply orientation rules to propagate edge directions
5. Hypothesis Evaluation: Check the specific hypothesis against the final causal graph

You must create stages that reconstruct the FULL causal graph, needed for accurate causal inference.

Input available in context: 'input' (contains premise with variables, correlations, conditional independencies, and hypothesis).
Final answer should be True or False.
"""

#     task_description = """
# Task: Given a natural-language input that contains a Premise and a Hypothesis, decide whether the Hypothesis is True or False under the Peter-Clark (PC) algorithm.

# - PC is a constraint-based causal discovery method that infers a causal equivalence class (CPDAG) from observational (in)dependence information.
# - Before deciding, reconstruct a global causal structure over all variables mentioned in the Premise; do NOT rely on pairwise or local checks.
# - Return True only if the claim holds in every DAG in the Markov equivalence class implied by the Premise; otherwise return False.

# Your plan must mirror the canonical PC algorithm. Reconstruct a global causal structure over all variables before deciding. Do not base the decision on a single pair or local cues. If your plan deviates from PC semantics, it is invalid.

# Input available in context: 'input' (contains premise with variables, correlations, conditional independencies, and hypothesis).
# Final answer should be True or False.
# """

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

    # Validate and refine schemas
    print(f"\n🔍 === SCHEMA VALIDATION & REFINEMENT ===")
    refined_stages = []

    for i, stage in enumerate(plan.stages, 1):
        print(f"\n🔢 Checking Stage {i}: {stage.id}")

        if is_schema_too_generic(stage.output_schema):
            print(f"  ⚠️  Schema is too generic: {stage.output_schema}")
            print(f"  🔄 Refining schema...")

            try:
                refined_schema = await refine_schema(stage.id, stage.writes, stage.prompt_template)
                stage.output_schema = refined_schema
                print(f"  ✅ Refined schema: {json.dumps(refined_schema, indent=2)}")
            except Exception as e:
                print(f"  ❌ Failed to refine schema: {e}")
                print(f"  📝 Using fallback schema")
        else:
            print(f"  ✅ Schema looks good: {stage.output_schema}")

        refined_stages.append(stage)

    # Update plan with refined stages
    plan.stages = refined_stages
    print(f"\n🎉 Schema validation complete!")

    print(f"\n📋 === DETAILED PLAN BREAKDOWN ===")
    for i, stage in enumerate(plan.stages, 1):
        print(f"\n🔢 STAGE {i}: {stage.id}")
        print(f"  📥 Reads from context: {stage.reads}")
        print(f"  📤 Writes to context: {stage.writes}")
        print(f"  📝 Prompt template: {stage.prompt_template}")
        print(f"  🏗️  Output schema: {json.dumps(stage.output_schema, indent=4)}")

    print(f"\n🌊 === CONTEXT FLOW ANALYSIS ===")
    all_keys = set(["input"])  # Start with initial context
    print(f"🏁 Initial context: {list(all_keys)}")

    for i, stage in enumerate(plan.stages, 1):
        print(f"\n🔢 After stage {i} ({stage.id}):")
        for key in stage.writes:
            all_keys.add(key)
        print(f"  📋 Available keys: {sorted(list(all_keys))}")

        # Check if this stage can read what it needs
        missing_reads = set(stage.reads) - all_keys
        if missing_reads:
            print(f"  ⚠️  WARNING: Stage {i} tries to read non-existent keys: {missing_reads}")

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
    print(f"Expected: {sample.get('label', 'Unknown')}")

    return {
        "plan": plan_dict,
        "final_context": final_context,
        "final_result": final_result,
        "expected": sample.get('label', 'Unknown')
    }


async def main() -> None:
    csv_path = "../data/test_dataset.csv"
    sample = fetch_sample(csv_path)
    result = await run_complete_workflow(sample)


if __name__ == "__main__":
    asyncio.run(main())
