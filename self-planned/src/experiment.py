import asyncio
import json
import pandas as pd
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .planner import create_planner, is_schema_too_generic, refine_schema
from .executor import run_plan

load_dotenv()


class ExperimentRunner:
    """Run experiments on multiple samples and collect detailed results."""

    def __init__(self, csv_path: str, output_dir: str = "experiments"):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset and return it."""
        df = pd.read_csv(self.csv_path)
        print(f"📊 Dataset loaded: {len(df)} samples")
        return df

    def sample_data(self, df: pd.DataFrame, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Sample n_samples from the dataset."""
        if seed is not None:
            random.seed(seed)

        if n_samples >= len(df):
            print(f"⚠️  Requested {n_samples} samples but dataset only has {len(df)}. Using all samples.")
            return df

        sampled_indices = random.sample(range(len(df)), n_samples)
        sampled_df = df.iloc[sampled_indices].copy()
        sampled_df = sampled_df.reset_index(drop=True)

        print(f"🎲 Sampled {len(sampled_df)} samples with seed {seed}")
        return sampled_df

    async def run_single_sample(self, sample: pd.Series, sample_idx: int) -> Dict[str, Any]:
        """Run the complete workflow on a single sample and return detailed results."""

        print(f"\n🔄 === PROCESSING SAMPLE {sample_idx} ===")
        print(f"Input: {sample['input'][:100]}...")
        print(f"Expected: {sample['label']}")

        start_time = time.time()

        try:
            # Create planner
            planner = create_planner()

            # Task description (using the current version)
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

            # Generate plan
            plan_start = time.time()
            result = await planner.run(task_description)
            plan = result.output
            plan_time = time.time() - plan_start

            print(f"✅ Plan generated: {len(plan.stages)} stages in {plan_time:.2f}s")

            # Validate and refine schemas
            schema_start = time.time()
            refined_count = 0

            for stage in plan.stages:
                if is_schema_too_generic(stage.output_schema):
                    try:
                        refined_schema = await refine_schema(stage.id, stage.writes, stage.prompt_template)
                        stage.output_schema = refined_schema
                        refined_count += 1
                    except Exception as e:
                        print(f"⚠️  Failed to refine schema for stage {stage.id}: {e}")

            schema_time = time.time() - schema_start
            print(f"🔍 Schema refinement: {refined_count}/{len(plan.stages)} stages refined in {schema_time:.2f}s")

            # Execute plan
            exec_start = time.time()
            initial_context = {"input": sample['input']}
            final_context = await run_plan(plan, initial_context)
            exec_time = time.time() - exec_start

            # Extract result
            final_key = plan.final_key or "hypothesis_result"
            final_result = final_context.get(final_key)

            total_time = time.time() - start_time

            # Determine success/correctness
            expected = sample['label']

            # Try to extract boolean from result
            predicted = None
            if isinstance(final_result, bool):
                predicted = final_result
            elif isinstance(final_result, dict):
                # Look for boolean values in the dict
                for key, value in final_result.items():
                    if isinstance(value, bool):
                        predicted = value
                        break
            elif isinstance(final_result, str):
                if final_result.lower() in ['true', '1', 'yes']:
                    predicted = True
                elif final_result.lower() in ['false', '0', 'no']:
                    predicted = False

            # Convert to 0/1 for comparison
            if predicted is not None:
                predicted_label = 1 if predicted else 0
                correct = predicted_label == expected
            else:
                predicted_label = None
                correct = False

            print(f"🎯 Result: {final_result} -> {predicted_label} (expected: {expected}) {'✅' if correct else '❌'}")

            return {
                "sample_idx": sample_idx,
                "success": True,
                "correct": correct,
                "expected_label": expected,
                "predicted_label": predicted_label,
                "raw_result": final_result,
                "total_time": total_time,
                "plan_time": plan_time,
                "schema_time": schema_time,
                "exec_time": exec_time,
                "num_stages": len(plan.stages),
                "schemas_refined": refined_count,
                "final_key": final_key,
                "num_variables": sample.get('num_variables', None),
                "template": sample.get('template', None),
                "plan_json": plan.model_dump(),
                "final_context": final_context,
                "error": None
            }

        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ ERROR: {e}")

            return {
                "sample_idx": sample_idx,
                "success": False,
                "correct": False,
                "expected_label": sample['label'],
                "predicted_label": None,
                "raw_result": None,
                "total_time": error_time,
                "plan_time": None,
                "schema_time": None,
                "exec_time": None,
                "num_stages": None,
                "schemas_refined": None,
                "final_key": None,
                "num_variables": sample.get('num_variables', None),
                "template": sample.get('template', None),
                "plan_json": None,
                "final_context": None,
                "error": str(e)
            }

    async def run_experiment(self, n_samples: int, seed: Optional[int] = None,
                           experiment_name: Optional[str] = None) -> str:
        """Run experiment on n_samples and save results."""

        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"🧪 === STARTING EXPERIMENT: {experiment_name} ===")
        print(f"📊 Samples: {n_samples}")
        print(f"🎲 Seed: {seed}")

        # Load and sample data
        df = self.load_dataset()
        sampled_df = self.sample_data(df, n_samples, seed)

        # Setup results file for incremental saving
        results_file = self.output_dir / f"{experiment_name}_results.csv"

        # Run experiment
        results = []
        for idx, sample in sampled_df.iterrows():
            result = await self.run_single_sample(sample, idx)
            results.append(result)

            # Save result immediately to CSV
            result_df = pd.DataFrame([result])
            if idx == 0:  # First sample - create new file with header
                result_df.to_csv(results_file, index=False, mode='w')
            else:  # Append without header
                result_df.to_csv(results_file, index=False, mode='a', header=False)

            print(f"💾 Sample {idx} result saved to {results_file}")

        # Create final results dataframe for summary calculations
        results_df = pd.DataFrame(results)

        # Calculate summary stats
        success_rate = results_df['success'].mean()
        accuracy = results_df['correct'].mean()
        avg_time = results_df['total_time'].mean()
        avg_stages = results_df['num_stages'].mean()

        print(f"\n📈 === EXPERIMENT RESULTS ===")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Avg Time: {avg_time:.2f}s")
        print(f"Avg Stages: {avg_stages:.1f}")

        # Results already saved incrementally, no need to save again

        # Save summary
        summary = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "seed": seed,
            "success_rate": success_rate,
            "accuracy": accuracy,
            "avg_time": avg_time,
            "avg_stages": avg_stages,
            "results_file": str(results_file)
        }

        summary_file = self.output_dir / f"{experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Results saved to: {results_file}")
        print(f"💾 Summary saved to: {summary_file}")

        return str(results_file)


async def main():
    """Example usage of the experiment runner."""

    runner = ExperimentRunner("../data/test_dataset.csv")

    # Run a small experiment
    results_file = await runner.run_experiment(
        n_samples=10,
        seed=42,
        experiment_name="test_run"
    )

    print(f"\n🎉 Experiment complete! Results: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())