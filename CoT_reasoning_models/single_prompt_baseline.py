import os
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pandas import DataFrame
from tqdm import tqdm

from answer_extractor import extract_premise, extract_hypothesis, extract_hypothesis_answer, compare_hypothesis_answers, \
    aggregate_metrics_single_prompt, display_metrics


def load_yaml(file_path: str) -> dict[str, str]:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters:
    - file_path (str): The path to the YAML file.

    Returns:
    - dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

PROMPTS: dict[str, str] = load_yaml("prompts.yaml")

print("Available prompt keys:", list(PROMPTS.keys()))


def run_single_experiment(client: OpenAI, row: pd.Series, max_attempts: int = 3) -> tuple[Optional[dict], dict]:
    # Create a log entry
    log_entry = row.to_dict()
    log_entry["sampleId"] = row.name
    log_entry["model_answer"] = None
    log_entry["model_label"] = None
    log_entry["attempt_count"] = 0

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        log_entry["attempt_count"] = attempts
        try:
            # Draw sample
            premise = extract_premise(row["input"])
            hypothesis = extract_hypothesis(row["input"])

            # Prepare the prompt
            prompt_template = PROMPTS["single_stage_prompt"]
            prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)
            print(f"\nInput prompt:\n{prompt}")

            # Ground truth label for given hypothesis
            label_int = row["label"]
            label_bool = bool(label_int)
            print(f"\nGround truth label:\n{label_bool}")

            # Generate the answer using the OpenAI API
            completion = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            # Model response
            model_response = completion.choices[0].message.content
            print(f"\nModel Response:\n{model_response}")
            log_entry["model_answer"] = model_response

            # Extract edges from the model response
            answer_label = extract_hypothesis_answer(answer=model_response)
            print(f"\nModel label:\n{answer_label}")
            log_entry["model_label"] = answer_label

            # Compare the expected edges with the model's predicted edges
            result = compare_hypothesis_answers(expected_answer=label_bool, model_answer=answer_label)
            print(f"\nComparison result:\n{result}")

            return result, log_entry
        except Exception as e:
            if attempts < max_attempts:
                print(f"Experiment failed on attempt {attempts}/{max_attempts}: {e}. Retrying...")
                continue
            else:
                print(f"Experiment failed after {max_attempts} attempts. Last error: {e}")
                return None, log_entry

    # This line should theoretically never be reached
    return None, log_entry


def run_multiple_experiments(client: OpenAI, df: DataFrame, num_experiments: int, log_file: str) -> None:
    results = []
    failed_experiments = 0

    num_samples = min(num_experiments, len(df))
    sampled_rows = df.sample(n=num_samples, replace=False)

    for _, row in tqdm(sampled_rows.iterrows(), total=num_samples, desc="Running Experiments"):
        result, log_entry = run_single_experiment(client=client, row=row)

        # Append the log entry to the CSV file
        log_df = pd.DataFrame([log_entry])
        if not os.path.exists(log_file):
            log_df.to_csv(log_file, index=False, mode='w')
        else:
            log_df.to_csv(log_file, index=False, header=False, mode='a')

        if result:
            results.append(result)
        else:
            failed_experiments += 1

    if results:
        aggregated_metrics = aggregate_metrics_single_prompt(results)
        display_metrics(aggregated_metrics)

    print(f"\nFailed  experiments: {failed_experiments}")


def main():
    NUM_EXPERIMENTS: int = 2

    # Retrieve the API key from the .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")

    # Load the dataframe
    csv_file_path = "data/test/test_dataset_unbalanced.csv"
    df = pd.read_csv(csv_file_path)

    # Initialize the API client
    client = OpenAI(
        api_key=api_key,
    )

    # Create a log directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "causal_discovery_single_prompt_experiment_logs.csv")

    # Run multiple experiments
    run_multiple_experiments(
        client=client,
        df=df,
        num_experiments=NUM_EXPERIMENTS,
        log_file=log_file,
    )


if __name__ == "__main__":
    main()
