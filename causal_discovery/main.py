import argparse
import datetime
import logging
import os
import time

import pandas as pd
from tqdm import tqdm

from utils import extract_premise, extract_hypothesis
from pipeline.pipeline import CausalDiscoveryPipeline, BatchCasualDiscoveryPipeline
from pipeline.stages import UndirectedSkeletonStage, VStructuresStage, MeekRulesStage, HypothesisEvaluationStage
from llm_client import OpenAIClient, BaseLLMClient, HuggingFaceClient


LOGS_DIR = "logs"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Causal Discovery Pipeline with configurable backend and mode."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "huggingface"],
        default="openai",
        help="Choose the LLM backend.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "batched"],
        default="sequential",
        help="Run pipeline in sequential or batched mode.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["normal", "balanced"],
        default="normal",
        help="Which test dataset to run. For example, 'normal' maps to data/test_dataset_unbalanced.csv, and 'balanced' maps to data/test_dataset_balanced.csv.",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="(Random mode only) Number of experiments to run. If greater than dataset length, the whole test set will be used.",
    )
    parser.add_argument(
        "--sample_method",
        type=str,
        choices=["random", "range"],
        default="random",
        help="Choose whether to sample randomly or use a defined range from the test set.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="(Range mode) Start index for the test set (inclusive).",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="(Range mode) End index for the test set (exclusive). If not provided, processing will continue to the end.",
    )
    return parser.parse_args()


def load_dataset(dataset_choice: str) -> pd.DataFrame:
    if dataset_choice == "balanced":
        csv_file = "data/test_dataset_balanced.csv"
    else:
        csv_file = "data/test_dataset_unbalanced.csv"
    df = pd.read_csv(csv_file)
    logging.info(f"Loaded dataset from {csv_file} with {len(df)} rows.")
    return df


def prepare_input_samples(df: pd.DataFrame, num_experiments: int, sample_method: str, start_index: int, end_index: int = None) -> list[dict]:
    if sample_method == "random":
        num_experiments = min(num_experiments, len(df))
        sampled_df = df.sample(n=num_experiments, replace=False)
    elif sample_method == "range":
        if end_index is None:
            sampled_df = df.iloc[start_index:]
        else:
            sampled_df = df.iloc[start_index:end_index]
    else:
        raise ValueError("Invalid sample_method provided.")

    input_samples = []
    for idx, row in sampled_df.iterrows():
        input_text = row["input"]
        premise = extract_premise(input_text)
        hypothesis = extract_hypothesis(input_text)
        sample = {
            "sample_id": idx,
            "sample_input": input_text,
            "sample_label": row["label"],
            "sample_num_variables": row["num_variables"],
            "sample_template": row["template"],
            "premise": premise,
            "hypothesis": hypothesis
        }
        input_samples.append(sample)
    logging.info(f"Prepared {len(input_samples)} input samples for the pipeline.")
    return input_samples


def create_client(backend: str) -> BaseLLMClient:
    if backend == "openai":
        client = OpenAIClient(model_id="o3-mini")
    else:
        client = HuggingFaceClient(max_new_tokens=8192, batch_size=4)
    logging.info(f"Using {backend} backend for the pipeline.")
    return client


def ensure_logs_directory_exists() -> None:
    """Create the logs directory if it doesn't already exist."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        logging.info(f"Created logs directory: {LOGS_DIR}")


def log_results(results: list[dict]) -> str:
    """
    Write the list of result dictionaries to a uniquely-named CSV log file in the logs directory.
    Assumes that each result dictionary contains a 'hypothesis_label' that is convertible to int.
    The column 'hypothesis_label' will be converted to numeric (0 or 1).
    """
    ensure_logs_directory_exists()

    # Create a unique filename using a timestamp.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join(LOGS_DIR, f"experiment_results_{timestamp}.csv")

    df_log = pd.DataFrame(results)
    df_log["hypothesis_label"] = df_log["hypothesis_label"].apply(lambda x: int(x))

    df_log.to_csv(log_file, index=False)
    logging.info(f"Logged results to {log_file}")
    return log_file


def post_process_logs(log_file: str) -> None:
    """
    Read the log CSV file, compute confusion matrix and performance metrics,
    then print them out.
    Assumes that each result dictionary contains:
      - "hypothesis_label": a dict with key "hypothesis_answer" (the model's prediction, boolean)
      - "sample_label": the ground truth label (boolean)
    """
    df = pd.read_csv(log_file)
    df["hypothesis_label"] = df["hypothesis_label"].astype(int)
    df["sample_label"] = df["sample_label"].astype(int)

    tp = ((df["hypothesis_label"] == 1) & (df["sample_label"] == 1)).sum()
    tn = ((df["hypothesis_label"] == 0) & (df["sample_label"] == 0)).sum()
    fp = ((df["hypothesis_label"] == 1) & (df["sample_label"] == 0)).sum()
    fn = ((df["hypothesis_label"] == 0) & (df["sample_label"] == 1)).sum()

    # Calculate metrics.
    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Confusion Matrix ---")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("\n--- Performance Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_arguments()

    # Load dataset and prepare input samples.
    df = load_dataset(args.dataset)
    input_samples = prepare_input_samples(df, args.num_experiments, args.sample_method, args.start_index, args.end_index)

    # Create the LLM client based on backend choice.
    client = create_client(args.backend)

    # Prepare the pipeline
    skeleton_stage = UndirectedSkeletonStage(client=client)
    v_structures_stage = VStructuresStage(client=client)
    meek_rules_stage = MeekRulesStage(client=client)
    hypothesis_evaluation_stage = HypothesisEvaluationStage(client=client)

    pipeline: CausalDiscoveryPipeline = CausalDiscoveryPipeline(
        stages=[skeleton_stage, v_structures_stage, meek_rules_stage, hypothesis_evaluation_stage]
    )

    results = []
    failed_ids = []
    start_time = time.time()

    if args.mode == "batched":
        logging.info("Running pipeline in batched mode.")
        batch_pipeline = BatchCasualDiscoveryPipeline(pipeline=pipeline)
        results, failed_ids = batch_pipeline.run_batch(input_samples)
    else:
        logging.info("Running pipeline in sequential mode.")
        for sample in tqdm(input_samples, desc="Processing samples"):
            try:
                result = pipeline.run(sample)
                results.append(result)
            except Exception as e:
                failed_ids.append(sample["sample_id"])
                logging.error(f"Error processing sample {sample['sample_id']}: {e}")

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Extend each result with additional sample metadata.
    for sample, res in zip(input_samples, results):
        res["sample_id"] = sample.get("sample_id")
        res["sample_input"] = sample.get("sample_input")
        res["sample_label"] = sample.get("sample_label")
        res["sample_num_variables"] = sample.get("sample_num_variables")
        res["sample_template"] = sample.get("sample_template")

    # Log results to CSV file.
    log_file = log_results(results)

    # Run results post-processing.
    post_process_logs(log_file)

    if failed_ids:
        logging.info(f"Total failed experiments after max retries: {len(failed_ids)}")
        logging.info(f"Failed sample IDs: {failed_ids}")


if __name__  == "__main__":
    main()
