import logging
from typing import Optional

import asyncio
from pandas import DataFrame
from openai import AsyncOpenAI

from answer_extractor import (
    extract_hypothesis_answer,
    compare_hypothesis_answers,
    aggregate_metrics_single_prompt,
    display_metrics,
)
from utils import prepare_experiment_from_row, append_log


_CONCURRENCY: int = 40

async def _run_one(
    client: AsyncOpenAI,
    experiment: dict,
    max_attempts: int = 3
) -> tuple[Optional[dict], dict]:
    for attempt in range(1, max_attempts + 1):
        experiment["attempt_count"] = attempt

        try:
            logging.info(f"[Sample {experiment['sampleId']}] Prompt:\n{experiment['prompt']}")
            resp = await client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": experiment["prompt"]}],
                temperature=0.1,
                stream=False,
            )

            # pull out usage & content
            usage = resp.usage
            experiment.update({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "model_answer": resp.choices[0].message.content,
            })
            logging.info(f"Model response:\n{experiment['model_answer']}")

            # extract & compare
            ans = extract_hypothesis_answer(experiment["model_answer"])
            experiment["model_label"] = int(ans)
            result = compare_hypothesis_answers(
                expected_answer=experiment["label"],
                model_answer=ans
            )
            logging.info(f"Comparison result: {result}")

            return result, experiment
        except Exception as e:
            logging.error(f"[Sample {experiment['sampleId']}] Attempt {attempt} failed: {e}", exc_info=True)
            if attempt == max_attempts:
                return None, experiment

    return None, experiment

async def run_experiments_deepseek_async(
    df: DataFrame,
    num_experiments: int,
    log_file: str,
    api_key: str,
) -> None:
    # prepare all experiments up front
    samples = df.sample(n=min(num_experiments, len(df)), replace=False)
    experiments = [prepare_experiment_from_row(row) for _, row in samples.iterrows()]

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    sem = asyncio.Semaphore(_CONCURRENCY)
    tasks = []

    async def sem_task(exp):
        async with sem:
            result, log_entry = await _run_one(client, exp)
            append_log(log_file, log_entry)
            return result

    for exp in experiments:
        tasks.append(asyncio.create_task(sem_task(exp)))

    results = await asyncio.gather(*tasks)

    # filter out failures (None)
    successes = [r for r in results if r is not None]
    if successes:
        metrics = aggregate_metrics_single_prompt(successes)
        display_metrics(metrics)

    logging.info(f"DeepSeek async done.  Total: {len(experiments)}, Succeeded: {len(successes)}")