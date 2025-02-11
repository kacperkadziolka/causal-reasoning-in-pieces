import os
from typing import Optional

import pandas as pd
from tqdm import tqdm

from ToT.agent import CausalSkeletonToT
from ToT.text_processor import extract_edges, compare_edges, aggregate_metrics
from ToT.utils import prompts, config


def load_df(file_path: str) -> pd.DataFrame:
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, file_path)
    return pd.read_csv(csv_file_path)


def draw_sample(df: pd.DataFrame, random_state: Optional[int] = None) -> tuple[str, str]:
    sample = df.sample(1, random_state=random_state)
    row_input = sample["input"].item()
    premise = row_input.split("Hypothesis:")[0].strip()
    row_expected_answer = sample["expected_answer"].item()
    return premise, row_expected_answer


def run_single_experiment(df: pd.DataFrame, agent: CausalSkeletonToT) -> dict[str, any]:
    premise, expected_answer = draw_sample(df)
    expected_edges = extract_edges(expected_answer)

    initial_prompt: str = prompts["initial_prompt"].format(premise=premise)
    final_output: str = agent.run(initial_prompt)

    if not final_output:
        raise ValueError("No output generated.")

    print(f"\nModel output:\n{final_output}")
    generated_edges = extract_edges(final_output)

    if generated_edges:
        return compare_edges(expected_edges, generated_edges)
    else:
        print("========================================")
        print(final_output)
        print("========================================")
        raise ValueError("Cannot parse the graph from model output.")


def run_experiments(df: pd.DataFrame, agent: CausalSkeletonToT, num_experiments: int) -> None:
    results = []
    failed_experiments = 0

    for _ in tqdm(range(num_experiments), desc="Running Experiments"):
        try:
            result = run_single_experiment(df, agent)
            results.append(result)
        except Exception as e:
            print(f"Experiment failed: {e}")
            failed_experiments += 1

    if results:
        print('\nExperiment finished. Results:')
        print(aggregate_metrics(results))

    print(f"Total failed experiments: {failed_experiments} out of {num_experiments}")


def main() -> None:
    # Load configuration
    data = config.get("data_path")
    num_experiments = config.get("num_experiments")
    max_steps = config.get("max_steps")
    threshold = config.get("threshold")

    df = load_df(data)
    agent: CausalSkeletonToT = CausalSkeletonToT(
        max_steps=max_steps,
        threshold=threshold
    )

    # Test: run single experiment
    # run_single_experiment(df, agent)

    run_experiments(
        df=df,
        agent=agent,
        num_experiments=num_experiments
    )


if __name__ == "__main__":
    main()
