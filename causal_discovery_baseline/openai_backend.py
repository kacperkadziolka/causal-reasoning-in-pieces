import logging
from typing import Optional

import asyncio
from tqdm.asyncio import tqdm_asyncio
from pandas import DataFrame
from openai import AsyncOpenAI

from answer_extractor import (
    extract_hypothesis_answer,
    compare_hypothesis_answers,
    aggregate_metrics_single_prompt,
    display_metrics,
)
from utils import prepare_experiment_from_row, append_log

_CONCURRENCY: int = 10


async def _run_one(
    client: AsyncOpenAI,
    experiment: dict,
    max_attempts: int = 3
) -> tuple[Optional[dict], dict]:
    for attempt in range(1, max_attempts + 1):
        experiment["attempt_count"] = attempt

        try:
            logging.info(f"[Sample {experiment['sample_id']}] Prompt:\n{experiment['prompt']}")
            logging.info(f"Ground truth label: {experiment['label']}")

            resp = await client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": experiment["prompt"]}],
            )

            # Extract token usage
            usage = resp.usage
            experiment["token_usage"]["input_tokens"] += usage.prompt_tokens
            experiment["token_usage"]["output_tokens"] += usage.completion_tokens
            experiment["token_usage"]["total_tokens"] += usage.total_tokens

            model_response = resp.choices[0].message.content
            experiment["model_answer"] = model_response
            logging.info(f"Model response:\n{model_response}")

            answer_label = extract_hypothesis_answer(answer=model_response)
            experiment["model_label"] = int(answer_label) if answer_label is not None else None
            logging.info(f"Model label: {experiment['model_label']}")

            result = compare_hypothesis_answers(
                expected_answer=experiment["label"],
                model_answer=answer_label
            )
            logging.info(f"Comparison result: {result}")

            return result, experiment
        except Exception as e:
            logging.error(f"[Sample {experiment['sample_id']}] Attempt {attempt} failed: {e}", exc_info=True)
            if attempt == max_attempts:
                return None, experiment

    return None, experiment


async def run_experiments_openai_async(
    df: DataFrame,
    num_experiments: int,
    log_file: str,
    api_key: str,
    prompt_type: str = "single_stage_prompt",
) -> None:
    # Prepare all experiments up front
    samples = df.sample(n=min(num_experiments, len(df)), replace=False)
    experiments = [prepare_experiment_from_row(row, prompt_type=prompt_type) for _, row in samples.iterrows()]

    client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def sem_task(exp):
        async with sem:
            result, log_entry = await _run_one(client, exp)
            append_log(log_file, log_entry)
            return result

    tasks = [asyncio.create_task(sem_task(exp)) for exp in experiments]

    results = await tqdm_asyncio.gather(*tasks, desc="Running OpenAI Experiments")

    # Filter out failures (None)
    successes = [r for r in results if r is not None]
    if successes:
        metrics = aggregate_metrics_single_prompt(successes)
        display_metrics(metrics)

    logging.info(f"OpenAI async done. Total: {len(experiments)}, Succeeded: {len(successes)}")
