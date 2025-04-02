import logging

from pandas import DataFrame
from tqdm import tqdm
from transformers import Pipeline

from answer_extractor import (
    extract_hypothesis_answer,
    compare_hypothesis_answers,
    aggregate_metrics_single_prompt,
    display_metrics,
)
from utils import prepare_experiment_from_row, append_log

# Huggingface configuration
BATCH_SIZE = 4
MAX_NEW_TOKENS = 8192


def process_hf_response(output_obj: dict, experiment: dict) -> bool:
    try:
        assistant_response = output_obj[0]["generated_text"][-1]["content"]
        experiment["model_answer"] = assistant_response
        logging.info(f"[Sample {experiment['sampleId']}] Model response:\n{assistant_response}")

        answer_label = extract_hypothesis_answer(answer=assistant_response)
        experiment["model_label"] = int(answer_label) if answer_label is not None else None
        experiment["attempt_count"] = 1

        result = compare_hypothesis_answers(
            expected_answer=experiment["expected_label"],
            model_answer=answer_label
        )
        logging.info(f"[Sample {experiment['sampleId']}] Comparison result: {result}")

        return result is not None
    except Exception as e:
        logging.error(f"[Sample {experiment['sampleId']}] Failed to process HF response: {e}", exc_info=True)
        return False


def run_experiments_hf(pipeline: Pipeline, df: DataFrame, num_experiments: int, log_file: str,
                       batch_size: int = BATCH_SIZE, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
    experiment_data = [
        prepare_experiment_from_row(row)
        for _, row in df.sample(n=min(num_experiments, len(df)), replace=False).iterrows()
    ]
    results = []
    retry_queue = []

    for i in tqdm(range(0, len(experiment_data), batch_size), desc="Running HF Batches"):
        batch = experiment_data[i:i + batch_size]
        batch_messages = [[{"role": "user", "content": exp["prompt"]}] for exp in batch]

        try:
            outputs = pipeline(
                batch_messages,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size
            )
        except Exception as e:
            logging.error(f"Pipeline error on batch starting at index {i}: {e}", exc_info=True)
            retry_queue.extend(batch)
            continue

        for output_obj, exp in zip(outputs, batch):
            if process_hf_response(output_obj, exp):
                results.append(exp)
            else:
                retry_queue.append(exp)
            append_log(log_file, exp)

    if retry_queue:
        logging.info(f"Retrying {len(retry_queue)} failed experiments in HF backend...")

        for i in tqdm(range(0, len(retry_queue), batch_size), desc="Retrying HF Batches"):
            retry_batch = retry_queue[i:i + batch_size]
            batch_messages = [[{"role": "user", "content": exp["prompt"]}] for exp in retry_batch]
            try:
                outputs = pipeline(
                    batch_messages,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size
                )
            except Exception as e:
                logging.error(f"Pipeline error on retry batch starting at index {i}: {e}", exc_info=True)
                continue
            for output_obj, exp in zip(outputs, retry_batch):
                if process_hf_response(output_obj, exp):
                    results.append(exp)
                else:
                    results.append({"exact_match": False})
                append_log(log_file, exp)

    if results:
        aggregated_metrics = aggregate_metrics_single_prompt(results)
        display_metrics(aggregated_metrics)

    logging.info(f"Total experiments processed (HF): {len(results)}; failed: {num_experiments - len(results)}")
