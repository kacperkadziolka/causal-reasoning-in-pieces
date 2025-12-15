import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from main import run_simple_workflow
from plan.models import Plan

load_dotenv()


@dataclass
class ExperimentConfig:
    """Configuration for batch experiments."""

    batch_size: int
    dataset_path: str = "../data/test_dataset.csv"
    output_dir: str = "../experiments"
    experiment_name: Optional[str] = None
    save_individual_results: bool = True
    save_summary: bool = True
    random_seed: Optional[int] = None
    max_concurrent: int = 10  # Maximum concurrent experiments
    sample_indices: Optional[List[int]] = None  # Specific sample indices to run
    use_sequential_generation: bool = False  # Sequential stage generation mode
    use_multi_agent_planner: bool = False  # Use MultiAgentPlanner instead of IterativePlanner
    use_plan_caching: bool = True  # Enable batched plan caching
    plan_cache_size: Optional[int] = None  # How many samples share a plan (default: max_concurrent)


@dataclass
class ExperimentResult:
    """Result from a single experiment."""

    sample_idx: int
    sample_input: str
    expected: bool
    predicted: bool
    is_correct: bool
    execution_time: float
    num_stages: int
    error: Optional[str] = None


@dataclass
class BatchResults:
    """Aggregated results from batch experiments."""

    config: ExperimentConfig
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_execution_time: float
    total_execution_time: float
    start_time: str
    end_time: str
    individual_results: List[ExperimentResult]


class BatchExperimentRunner:
    """Runs batch experiments with proper sampling and result aggregation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset: Optional[pd.DataFrame] = None
        self.results: List[ExperimentResult] = []
        self.start_time: Optional[float] = None

    def load_dataset(self) -> None:
        """Load and validate the dataset."""
        try:
            self.dataset = pd.read_csv(self.config.dataset_path)
            print(f"📊 Dataset loaded: {len(self.dataset)} samples")

            # Validate required columns
            required_cols = ['input', 'label']
            missing_cols = [col for col in required_cols if col not in self.dataset.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        except Exception as e:
            raise ValueError(f"Failed to load dataset from {self.config.dataset_path}: {e}")

    def validate_batch_size(self) -> int:
        """Validate and adjust batch size based on dataset size."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        dataset_size = len(self.dataset)
        requested_size = self.config.batch_size

        if requested_size > dataset_size:
            print(f"⚠️  Requested batch size ({requested_size}) exceeds dataset size ({dataset_size})")
            print(f"🔧 Adjusting batch size to dataset size: {dataset_size}")
            return dataset_size

        return requested_size

    def sample_indices(self, batch_size: int) -> List[int]:
        """Sample indices without replacement or use specific indices if provided."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # If specific sample indices are provided, use them (up to batch_size)
        if self.config.sample_indices is not None:
            available_indices = sorted(self.config.sample_indices)
            selected_indices = available_indices[:batch_size]  # Take only first batch_size indices
            print(f"📋 Using provided sample indices ({len(selected_indices)} samples from {len(available_indices)} available)")
            return selected_indices

        # Otherwise, sample randomly as before
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        total_samples = len(self.dataset)
        indices = np.random.choice(total_samples, size=batch_size, replace=False)
        return sorted(indices.tolist())

    async def run_single_experiment(
        self,
        sample_idx: int,
        cached_plan: Optional[Plan] = None,
        cached_knowledge: Optional[str] = None
    ) -> ExperimentResult:
        """Run a single experiment and return structured result.

        Args:
            sample_idx: Index of the sample in the dataset
            cached_plan: Optional pre-generated plan to skip planning phase
            cached_knowledge: Optional pre-extracted knowledge to skip extraction phase

        Returns:
            ExperimentResult with execution outcome and metrics
        """
        start_time = time.time()

        try:
            if self.dataset is None:
                raise ValueError("Dataset not loaded")
            sample = self.dataset.iloc[sample_idx]

            # Run the simple workflow with optional cached plan (verbose=False for batch mode)
            result = await run_simple_workflow(
                sample,
                use_sequential_generation=self.config.use_sequential_generation,
                use_multi_agent_planner=self.config.use_multi_agent_planner,
                cached_plan=cached_plan,
                cached_knowledge=cached_knowledge,
                verbose=False  # Suppress detailed logs in batch mode
            )
            execution_time = time.time() - start_time

            if result is None:
                return ExperimentResult(
                    sample_idx=sample_idx,
                    sample_input=sample['input'],
                    expected=bool(sample['label']),
                    predicted=False,  # Default for failed experiments
                    is_correct=False,
                    execution_time=execution_time,
                    num_stages=0,
                    error="Workflow failed to generate result"
                )

            return ExperimentResult(
                sample_idx=sample_idx,
                sample_input=sample['input'],
                expected=result['expected'],
                predicted=result['predicted'],
                is_correct=result['is_correct'],
                execution_time=execution_time,
                num_stages=result.get('num_stages', 0),
                error=None
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ExperimentResult(
                sample_idx=sample_idx,
                sample_input="Error loading sample",
                expected=False,
                predicted=False,
                is_correct=False,
                execution_time=execution_time,
                num_stages=0,
                error=str(e)
            )

    def print_progress(self, current: int, total: int, result: ExperimentResult) -> None:
        """Print progress information in condensed format."""
        progress_pct = (current / total) * 100
        status = "✅" if result.error is None else "❌"
        correct_symbol = "✓" if result.is_correct else "✗"

        # Format: [1/10] sample_idx | predicted=X expected=Y | ✓/✗ | 25.3s
        predicted_str = "T" if result.predicted else "F"
        expected_str = "T" if result.expected else "F"

        print(f"[{current:2d}/{total}] Sample {result.sample_idx:4d} | "
              f"pred={predicted_str} exp={expected_str} | {correct_symbol} | "
              f"{result.execution_time:5.1f}s {status}", flush=True)

    async def run_batch(self) -> BatchResults:
        """Run the complete batch experiment with plan caching."""
        print("🚀 Starting batch experiments...")

        # Load and validate dataset
        self.load_dataset()
        batch_size = self.validate_batch_size()

        # Sample indices
        sample_indices = self.sample_indices(batch_size)
        print(f"📋 Selected {len(sample_indices)} samples: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")

        # Determine cache size (samples per plan)
        cache_size = self.config.plan_cache_size or self.config.max_concurrent
        if not self.config.use_plan_caching:
            cache_size = 1  # Disable caching (one plan per sample)

        print(f"📦 Plan caching: {'ENABLED' if self.config.use_plan_caching else 'DISABLED'}")
        if self.config.use_plan_caching:
            num_groups = (len(sample_indices) + cache_size - 1) // cache_size
            print(f"   Cache size: {cache_size} samples per plan")
            print(f"   Batch groups: {num_groups}")

        # Plan cache - maps batch_group_id → (plan, knowledge)
        plan_cache: Dict[int, Tuple[Plan, str]] = {}
        plan_locks: Dict[int, asyncio.Lock] = {}

        async def generate_plan_for_group(batch_group_id: int) -> Tuple[Plan, str]:
            """Generate plan using first sample of the batch group."""
            import sys
            import io
            import time as time_module

            # Get first sample index in this group
            group_start_idx = batch_group_id * cache_size
            if group_start_idx >= len(sample_indices):
                raise ValueError(f"Invalid batch group {batch_group_id}")

            first_sample_idx = sample_indices[group_start_idx]
            first_sample = self.dataset.iloc[first_sample_idx]

            print(f"\n🔧 Plan {batch_group_id + 1}/{num_groups}: Generating for sample {first_sample_idx}...")

            plan_start = time_module.time()

            # Extract knowledge (current approach - uses sample)
            from knowledge.extractor import EnhancedKnowledgeExtractor
            extractor = EnhancedKnowledgeExtractor()
            knowledge = await extractor.extract_simple_knowledge(
                "Peter-Clark (PC) Algorithm",
                first_sample["input"]
            )

            # Generate plan
            from main import TASK_DESCRIPTION  # Import task description
            if self.config.use_multi_agent_planner:
                from plan.multi_agent_planner import MultiAgentPlanner
                planner = MultiAgentPlanner()
                plan, _ = await planner.generate_plan(
                    task_description=TASK_DESCRIPTION,
                    algorithm_knowledge=knowledge,
                    use_sequential=self.config.use_sequential_generation,
                    verbose=False  # Suppress detailed planning logs in batch mode
                )
            else:
                from plan.iterative_planner import IterativePlanner
                planner = IterativePlanner()
                plan = await planner.generate_two_stage_plan(
                    task_description=TASK_DESCRIPTION,
                    algorithm_knowledge=knowledge,
                    enhance_prompts=True
                )

            plan_time = time_module.time() - plan_start

            # Print condensed summary
            stage_names = [s.id for s in plan.stages]
            print(f"\n✅ Plan {batch_group_id + 1}/{num_groups} complete: {len(plan.stages)} stages [{', '.join(stage_names)}] ({plan_time:.1f}s)", flush=True)

            return plan, knowledge

        # Initialize timing
        self.start_time = time.time()
        start_datetime = datetime.now()

        print(f"\n⏳ Running {batch_size} experiments in batch groups...")
        print("=" * 80)

        # Process samples in batch groups sequentially
        all_results = []
        completed_count = 0
        num_groups = (len(sample_indices) + cache_size - 1) // cache_size

        for batch_group_id in range(num_groups):
            # Determine which samples belong to this batch group
            group_start_idx = batch_group_id * cache_size
            group_end_idx = min(group_start_idx + cache_size, len(sample_indices))
            group_sample_indices = sample_indices[group_start_idx:group_end_idx]

            # Generate plan for this batch group
            try:
                plan, knowledge = await generate_plan_for_group(batch_group_id)
                plan_cache[batch_group_id] = (plan, knowledge)
            except Exception as e:
                print(f"❌ Plan generation failed for group {batch_group_id + 1}/{num_groups}: {e}")
                # Create error results for all samples in this group
                for sample_idx in group_sample_indices:
                    error_result = ExperimentResult(
                        sample_idx=sample_idx,
                        sample_input="Plan generation failed",
                        expected=False,
                        predicted=False,
                        is_correct=False,
                        execution_time=0.0,
                        num_stages=0,
                        error=f"Plan generation failed: {e}"
                    )
                    all_results.append(error_result)
                    completed_count += 1
                continue

            # Execute samples in this group concurrently
            print(f"\n⚡ Executing {len(group_sample_indices)} samples from batch group {batch_group_id + 1}/{num_groups}...")
            print("-" * 80)

            semaphore = asyncio.Semaphore(self.config.max_concurrent)

            async def run_with_semaphore(sample_idx: int) -> ExperimentResult:
                nonlocal completed_count
                async with semaphore:
                    result = await self.run_single_experiment(
                        sample_idx,
                        cached_plan=plan,
                        cached_knowledge=knowledge
                    )
                    completed_count += 1
                    self.print_progress(completed_count, batch_size, result)
                    return result

            # Execute all samples in this group concurrently
            group_tasks = [run_with_semaphore(idx) for idx in group_sample_indices]
            group_results = await asyncio.gather(*group_tasks)
            all_results.extend(group_results)

        self.results = all_results

        # Calculate final metrics
        end_time = time.time()
        end_datetime = datetime.now()
        total_time = end_time - self.start_time

        successful_results = [r for r in self.results if r.error is None]
        accuracy = sum(r.is_correct for r in successful_results) / len(successful_results) if successful_results else 0.0
        avg_execution_time = float(np.mean([r.execution_time for r in self.results]))

        # Calculate precision, recall, and F1 score
        if successful_results:
            # For binary classification: True Positive, True Negative, False Positive, False Negative
            true_positives = sum(1 for r in successful_results if r.expected and r.predicted)
            false_positives = sum(1 for r in successful_results if not r.expected and r.predicted)
            false_negatives = sum(1 for r in successful_results if r.expected and not r.predicted)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1_score = 0.0

        batch_results = BatchResults(
            config=self.config,
            total_experiments=len(self.results),
            successful_experiments=len(successful_results),
            failed_experiments=len(self.results) - len(successful_results),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            avg_execution_time=avg_execution_time,
            total_execution_time=total_time,
            start_time=start_datetime.isoformat(),
            end_time=end_datetime.isoformat(),
            individual_results=self.results
        )

        return batch_results

    def save_results(self, batch_results: BatchResults) -> Tuple[str, str]:
        """Save experiment results to files."""
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.experiment_name or f"batch_exp_{timestamp}"

        # Save individual results as CSV
        results_file = None
        if self.config.save_individual_results:
            results_data = [asdict(result) for result in batch_results.individual_results]
            results_df = pd.DataFrame(results_data)
            results_file = output_dir / f"{exp_name}_results.csv"
            results_df.to_csv(results_file, index=False)

        # Save summary as JSON
        summary_file = None
        if self.config.save_summary:
            summary_data = asdict(batch_results)
            # Remove individual results from summary to keep it concise
            summary_data['individual_results'] = f"See {exp_name}_results.csv"
            summary_file = output_dir / f"{exp_name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)

        return str(results_file), str(summary_file)

    def print_summary(self, batch_results: BatchResults) -> None:
        """Print comprehensive experiment summary."""
        print("\n" + "=" * 80)
        print("📊 BATCH EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Total Experiments:     {batch_results.total_experiments}")
        print(f"Successful:            {batch_results.successful_experiments}")
        print(f"Failed:                {batch_results.failed_experiments}")
        print(f"Success Rate:          {(batch_results.successful_experiments/batch_results.total_experiments)*100:.1f}%")
        print(f"Accuracy:              {batch_results.accuracy*100:.1f}%")
        print(f"Precision:             {batch_results.precision*100:.1f}%")
        print(f"Recall:                {batch_results.recall*100:.1f}%")
        print(f"F1 Score:              {batch_results.f1_score*100:.1f}%")
        print(f"Avg Execution Time:    {batch_results.avg_execution_time:.2f}s")
        print(f"Total Execution Time:  {batch_results.total_execution_time:.1f}s")

        if batch_results.failed_experiments > 0:
            print("\n❌ Failed Experiments:")
            for result in batch_results.individual_results:
                if result.error:
                    print(f"   Sample {result.sample_idx}: {result.error}")

        print("=" * 80)


async def main():
    """Example usage of batch experiment runner."""
    config = ExperimentConfig(
        batch_size=5,  # Start small for testing
        random_seed=42,
        experiment_name="test_batch"
    )

    runner = BatchExperimentRunner(config)
    batch_results = await runner.run_batch()

    # Print summary
    runner.print_summary(batch_results)

    # Save results
    results_file, summary_file = runner.save_results(batch_results)
    print("\n💾 Results saved:")
    print(f"   📄 Details: {results_file}")
    print(f"   📋 Summary: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())