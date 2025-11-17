import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from main import run_enhanced_workflow

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
    # Enhanced logging fields
    plan_json: Optional[str] = None
    stage_details: Optional[str] = None
    stage_prompts: Optional[str] = None
    intermediate_outputs: Optional[str] = None
    final_context: Optional[str] = None


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
        """Sample indices without replacement."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        total_samples = len(self.dataset)
        indices = np.random.choice(total_samples, size=batch_size, replace=False)
        return sorted(indices.tolist())

    async def run_single_experiment(self, sample_idx: int) -> ExperimentResult:
        """Run a single experiment and return structured result."""
        start_time = time.time()

        try:
            if self.dataset is None:
                raise ValueError("Dataset not loaded")
            sample = self.dataset.iloc[sample_idx]

            # Run the experiment
            result = await run_enhanced_workflow(sample)
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
                    error="Workflow failed to generate result",
                    plan_json=None,
                    stage_details=None,
                    stage_prompts=None,
                    intermediate_outputs=None,
                    final_context=None
                )

            # Extract enhanced information
            plan_dict = result.get('plan_summary', {})
            final_context = result.get('final_context', {})

            # Serialize plan and stage information
            plan_json = json.dumps(plan_dict) if plan_dict else None

            # Extract stage details and prompts from enhanced workflow
            stage_details = []
            stage_prompts = []
            if 'stages' in plan_dict:
                for i, stage in enumerate(plan_dict['stages'], 1):
                    stage_details.append(f"Stage {i}: {stage.get('id', 'unknown')} | Reads: {stage.get('reads', [])} | Writes: {stage.get('writes', [])}")
                    prompt_template = stage.get('prompt_template', 'No prompt available')
                    # Keep full prompts for analysis - truncate only if extremely long
                    if len(prompt_template) > 2000:
                        truncated_prompt = prompt_template[:2000] + "... (truncated for length)"
                    else:
                        truncated_prompt = prompt_template
                    stage_prompts.append(f"Stage {i} Prompt:\n{truncated_prompt}\n")

            # Extract intermediate outputs (context keys and their values)
            intermediate_outputs = []
            for key, value in final_context.items():
                if key != 'input':  # Skip the input as it's already captured
                    intermediate_outputs.append(f"{key}: {str(value)}")

            return ExperimentResult(
                sample_idx=sample_idx,
                sample_input=sample['input'],
                expected=result['expected'],
                predicted=result['predicted'],
                is_correct=result['is_correct'],
                execution_time=execution_time,
                num_stages=len(result['plan_summary']['stages']) if 'plan_summary' in result else 0,
                error=None,
                plan_json=plan_json,
                stage_details="\n".join(stage_details) if stage_details else None,
                stage_prompts="\n".join(stage_prompts) if stage_prompts else None,
                intermediate_outputs="\n".join(intermediate_outputs) if intermediate_outputs else None,
                final_context=json.dumps(final_context) if final_context else None
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
                error=str(e),
                plan_json=None,
                stage_details=None,
                stage_prompts=None,
                intermediate_outputs=None,
                final_context=None
            )

    def print_progress(self, current: int, total: int, result: ExperimentResult) -> None:
        """Print progress information."""
        progress_pct = (current / total) * 100
        status = "✅" if result.error is None else "❌"
        correct = "✓" if result.is_correct else "✗"

        print(f"[{current:3d}/{total}] {progress_pct:5.1f}% {status} Sample {result.sample_idx:3d} "
              f"| {result.execution_time:5.1f}s | {correct} | {result.error or 'Success'}")

    async def run_batch(self) -> BatchResults:
        """Run the complete batch experiment."""
        print("🚀 Starting batch experiments...")

        # Load and validate dataset
        self.load_dataset()
        batch_size = self.validate_batch_size()

        # Sample indices
        sample_indices = self.sample_indices(batch_size)
        print(f"📋 Selected {len(sample_indices)} samples: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")

        # Initialize timing
        self.start_time = time.time()
        start_datetime = datetime.now()

        print(f"\n⏳ Running {batch_size} experiments concurrently (max {self.config.max_concurrent} at once)...")
        print("=" * 80)

        # Run experiments concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        completed_count = 0

        async def run_with_semaphore(sample_idx: int) -> ExperimentResult:
            nonlocal completed_count
            async with semaphore:
                result = await self.run_single_experiment(sample_idx)
                completed_count += 1
                self.print_progress(completed_count, batch_size, result)
                return result

        # Create all tasks and run them concurrently
        tasks = [run_with_semaphore(sample_idx) for sample_idx in sample_indices]
        self.results = await asyncio.gather(*tasks)

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