import logging
from typing import Optional

import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from answer_extractor import (
    extract_shortest_path_answer,
    compare_shortest_path,
    parse_graph_from_question,
    aggregate_metrics,
    display_metrics,
)
from utils import prepare_experiment_from_sample, append_log

_CONCURRENCY: int = 10


async def _run_one(
    client: AsyncOpenAI,
    experiment: dict,
    max_attempts: int = 3,
) -> tuple[Optional[dict], dict]:
    for attempt in range(1, max_attempts + 1):
        experiment["attempt_count"] = attempt

        try:
            logging.info(f"[Sample {experiment['sample_id']}] Prompt:\n{experiment['prompt'][:200]}...")
            logging.info(f"Ground truth weight: {experiment['ground_truth_weight']}")

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
            logging.info(f"Model response:\n{model_response[:500]}")

            # Extract path and weight from response
            extracted = extract_shortest_path_answer(model_response)
            experiment["model_path"] = extracted["path"]
            experiment["model_weight"] = extracted["weight"]
            logging.info(f"Extracted path: {extracted['path']}, weight: {extracted['weight']}")

            # Parse graph for path validation
            graph_info = parse_graph_from_question(experiment["question"])

            # Compare with ground truth
            result = compare_shortest_path(
                ground_truth_weight=experiment["ground_truth_weight"],
                model_path=extracted["path"],
                model_weight=extracted["weight"],
                graph_info=graph_info,
            )
            result["difficulty"] = experiment["difficulty"]
            logging.info(f"Comparison result: {result}")

            return result, experiment
        except Exception as e:
            logging.error(
                f"[Sample {experiment['sample_id']}] Attempt {attempt} failed: {e}",
                exc_info=True,
            )
            if attempt == max_attempts:
                return None, experiment

    return None, experiment


async def run_experiments_openai_async(
    samples: list[dict],
    num_experiments: int,
    log_file: str,
    api_key: str,
    prompt_type: str = "direct_prompt",
) -> None:
    import random

    # Select samples
    n = min(num_experiments, len(samples))
    selected = random.sample(samples, n) if n < len(samples) else samples
    experiments = [prepare_experiment_from_sample(s, prompt_type=prompt_type) for s in selected]

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
        difficulties = [r["difficulty"] for r in successes]
        metrics = aggregate_metrics(successes, difficulties)
        display_metrics(metrics)

    # Token usage summary
    total_input = sum(e["token_usage"]["input_tokens"] for e in experiments)
    total_output = sum(e["token_usage"]["output_tokens"] for e in experiments)
    total_tokens = sum(e["token_usage"]["total_tokens"] for e in experiments)
    n_run = len(experiments)

    print("\n" + "=" * 50)
    print("Token Usage Summary")
    print("=" * 50)
    print(f"Samples run: {n_run}")
    print(f"Input tokens:  {total_input:>10,}  (avg {total_input // n_run:,}/sample)")
    print(f"Output tokens: {total_output:>10,}  (avg {total_output // n_run:,}/sample)")
    print(f"Total tokens:  {total_tokens:>10,}  (avg {total_tokens // n_run:,}/sample)")

    # Cost estimate for full dataset
    total_samples = len(samples)
    if n_run < total_samples:
        est_input = total_input / n_run * total_samples
        est_output = total_output / n_run * total_samples
        # o3-mini pricing: $1.10/1M input, $4.40/1M output
        est_cost = (est_input / 1_000_000 * 1.10) + (est_output / 1_000_000 * 4.40)
        print(f"\nEstimated for full dataset ({total_samples} samples):")
        print(f"  Input tokens:  {est_input:>12,.0f}")
        print(f"  Output tokens: {est_output:>12,.0f}")
        print(f"  Est. cost (o3-mini): ${est_cost:.2f}")
    print("=" * 50)

    failures = len(results) - len(successes)
    logging.info(f"OpenAI async done. Total: {len(experiments)}, Succeeded: {len(successes)}, Failed: {failures}")
