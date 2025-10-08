import asyncio
from typing import Any, Optional
from dotenv import load_dotenv
import pandas as pd
import random
from planner import create_planner
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

Input data:
{sample['input']}

Inputs available in context: 'input'.
Final answer should be True or False.
"""

    print("Generating plan...")
    result = await planner.run(task_description)
    plan = result.output

    print(f"✓ Plan generated with {len(plan.stages)} stages")
    print(f"Final key: {plan.final_key}")

    print("\n=== GENERATED STAGES ===")
    for i, stage in enumerate(plan.stages, 1):
        print(f"\nStage {i}: {stage.id}")
        print(f"  Reads: {stage.reads}")
        print(f"  Writes: {stage.writes}")
        print(f"  Prompt: {stage.prompt_template}")

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
