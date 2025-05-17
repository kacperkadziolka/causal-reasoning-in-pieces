import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from causal_discovery.experiment_logger import ExperimentLogger
from utils import extract_premise, extract_hypothesis
from pipeline.pipeline import CausalDiscoveryPipeline, BatchCasualDiscoveryPipeline
from pipeline.stages import UndirectedSkeletonStage, VStructuresStage, MeekRulesStage, HypothesisEvaluationStage
from llm_client import OpenAIClient, BaseLLMClient, HuggingFaceClient, DeepSeekClient

LOGS_DIR: Path = Path("logs")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Causal Discovery Pipeline with configurable backend and mode."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the split csv file",
        default="../data/test_dataset_unbalanced.csv"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "huggingface", "deepseek"],
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
        "--num_experiments",
        type=int,
        default=1,
        help="Number of experiments to run. If greater than dataset length, the whole test set will be used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for batch processing.",
    )
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    csv_file = args.input_file
    df = pd.read_csv(csv_file)
    logging.info(f"Loaded dataset from {csv_file} with {len(df)} rows.")
    return df


def prepare_input_samples(df: pd.DataFrame, num_experiments: int) -> list[dict]:
    num_experiments = min(num_experiments, len(df))
    sampled_df = df.sample(n=num_experiments, replace=False)

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


def create_client(backend: str, batch_size: int) -> BaseLLMClient:
    if backend == "openai":
        client = OpenAIClient(model_id="o3-mini", concurrency=batch_size)
    elif backend == "huggingface":
        client = HuggingFaceClient(max_new_tokens=8192,  batch_size=batch_size, model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    else:
        client = DeepSeekClient(concurrency=batch_size)
    logging.info(f"Using {backend} backend for the pipeline.")
    return client


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
    args = parse_arguments()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load dataset and prepare input samples.
    df = load_dataset(args)
    input_samples = prepare_input_samples(df, args.num_experiments)

    # Create the LLM client based on backend choice.
    client = create_client(args.backend, args.batch_size)
    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    # Prepare the pipeline
    skeleton_stage = UndirectedSkeletonStage(client=client)
    v_structures_stage = VStructuresStage(client=client)
    meek_rules_stage = MeekRulesStage(client=client)
    hypothesis_evaluation_stage = HypothesisEvaluationStage(client=client)

    job_id = Path(args.input_file).stem
    logger = ExperimentLogger(LOGS_DIR, job_id)
    pipeline: CausalDiscoveryPipeline = CausalDiscoveryPipeline(
        stages=[skeleton_stage, v_structures_stage, meek_rules_stage, hypothesis_evaluation_stage],
        logger=logger,
    )

    results = []
    failed_ids = []
    start_time = time.time()

    if args.mode == "batched":
        logging.info("Running pipeline in batched mode.")
        batch_pipeline = BatchCasualDiscoveryPipeline(pipeline=pipeline, batch_size=args.batch_size)
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

    # Run results post-processing.
    post_process_logs(str(logger.log_file))

    if failed_ids:
        logging.info(f"Total failed experiments after max retries: {len(failed_ids)}")
        logging.info(f"Failed sample IDs: {failed_ids}")


if __name__  == "__main__":
    main()
