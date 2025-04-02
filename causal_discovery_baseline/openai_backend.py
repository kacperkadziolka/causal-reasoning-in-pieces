import logging
from typing import Tuple, Optional
from pandas import DataFrame
from tqdm import tqdm
from openai import OpenAI

from answer_extractor import (
    extract_hypothesis_answer,
    compare_hypothesis_answers,
    aggregate_metrics_single_prompt,
    display_metrics,
)
from utils import prepare_experiment_from_row, append_log


def run_openai_experiment(client: OpenAI, experiment: dict, max_attempts: int = 3) -> Tuple[Optional[dict], dict]:
    for attempt in range(1, max_attempts + 1):
        experiment["attempt_count"] = attempt

        try:
            logging.info(f"[Sample {experiment['sampleId']}] Prompt:\n{experiment['prompt']}")
            logging.info(f"Ground truth label: {experiment['expected_label']}")

            completion = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": experiment["prompt"]}]
            )

            model_response = completion.choices[0].message.content
            experiment["model_answer"] = model_response
            logging.info(f"Model response:\n{model_response}")

            answer_label = extract_hypothesis_answer(answer=model_response)
            experiment["model_label"] = int(answer_label) if answer_label is not None else None
            logging.info(f"Model label: {experiment['model_label']}")

            result = compare_hypothesis_answers(
                expected_answer=experiment["expected_label"],
                model_answer=answer_label
            )
            logging.info(f"Comparison result: {result}")

            return result, experiment
        except Exception as e:
            logging.error(f"[Sample {experiment['sampleId']}] Attempt {attempt} failed: {e}", exc_info=True)
            if attempt == max_attempts:
                return None, experiment

    return None, experiment


def run_experiments_openai(client: OpenAI, df: DataFrame, num_experiments: int, log_file: str) -> None:
    results = []
    num_samples = min(num_experiments, len(df))
    sampled_rows = df.sample(n=num_samples, replace=False)

    for _, row in tqdm(sampled_rows.iterrows(), total=num_samples, desc="Running OpenAI Experiments"):
        experiment = prepare_experiment_from_row(row)
        result, log_entry = run_openai_experiment(client, experiment)
        append_log(log_file, log_entry)

        if result:
            results.append(result)

    if results:
        aggregated_metrics = aggregate_metrics_single_prompt(results)
        display_metrics(aggregated_metrics)

    logging.info(f"Total failed experiments: {num_samples - len(results)}")
