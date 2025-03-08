import csv
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pandas import DataFrame
from tqdm import tqdm

from answer_extractor import aggregate_metrics, display_metrics, extract_edges_incident_format, extract_premise, \
    extract_edges_json, compare_edges

CSV_FIELDS: list[str] = [
    "input_prompt",
    "expected_answer",
    "model_answer",
    "expected_edges",
    "model_edges",
    "missing_edges",
    "extra_edges"
]
LOGS_DIR: str = "logs"


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


def ensure_logs_directory_exists():
    """
    Create the logs directory if it doesn't exist.
    """
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        print(f"Created logs directory: {LOGS_DIR}")


def get_log_filenames(temperature: int, do_sample: bool, num_experiments: int, no_variables: int) -> dict:
    """
    Generate log filenames based on configuration parameters.

    :param temperature: Temperature value used for generation
    :param do_sample: Whether sampling was used
    :param num_experiments: Number of experiments
    :param no_variables: Number of variables in the experiment
    :return: Dictionary with filenames for successful and failed experiments
    """
    # Ensure logs directory exists
    ensure_logs_directory_exists()

    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%d-%b-%Y-%H%M")

    # Build the base filename
    base_filename = f"{date_str}-temp{temperature}"

    if do_sample:
        base_filename += "-dosample"

    base_filename += f"-{num_experiments}exp-{no_variables}var"

    return {
        "successful": os.path.join(LOGS_DIR, f"{base_filename}-successful.csv"),
        "failed": os.path.join(LOGS_DIR, f"{base_filename}-failed.csv")
    }


def log_experiment_csv(entry: dict, file_path: str) -> None:
    """
    Log an experiment to the appropriate CSV file.

    :param entry: Dictionary containing the experiment data.
    :param file_path: Path to the CSV log file.
    """
    entry["expected_edges"] = str(list(entry["expected_edges"]))
    entry["model_edges"] = str(list(entry["model_edges"]))
    entry["missing_edges"] = str(list(entry["missing_edges"]))
    entry["extra_edges"] = str(list(entry["extra_edges"]))

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


def run_single_experiment(client: OpenAI, df: DataFrame, log_file_names: dict) -> Optional[dict]:
    """
    Run a single experiment to compare expected edges with the model's predicted edges.

    :param log_file_names: Dictionary containing the filenames for successful and failed experiments.
    :param client: OpenAI API client.
    :param df: DataFrame containing the questions and expected answers.
    :return: A dictionary with the comparison result.
    """
    try:
        # Draw sample
        sample = df.sample(n=1)
        premise = extract_premise(sample["input"].iloc[0])
        print(f"\nSelected a sample No. {sample.index[0]}.")

        # Prepare the prompt
        prompt_template = PROMPTS["reasoning_prompt"]
        prompt = prompt_template.format(premise=premise)
        print(f"\nInput prompt:\n{prompt}")

        # Expected answer
        expected_answer = sample["expected_answer"].iloc[0]
        expected_edges = extract_edges_incident_format(answer=expected_answer, step=5)
        print(f"\nExpected edges:\n{expected_edges}")

        # Generate the answer using the OpenAI API
        completion = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Model response
        model_response = completion.choices[0].message.content
        print(f"\nModel Response:\n{model_response}")

        # Extract edges from the model response
        answer_edges = extract_edges_json(answer=model_response)
        print(f"\nModel edges:\n{answer_edges}")

        # Compare the expected edges with the model's predicted edges
        result = compare_edges(expected_edges, answer_edges)
        print(f"\nComparison result:\n{result}")

        csv_log_entry = {
            "input_prompt": prompt,
            "expected_answer": expected_answer,
            "model_answer": model_response,
            "expected_edges": expected_edges,
            "model_edges": answer_edges,
            "missing_edges": result["missing_edges"],
            "extra_edges": result["extra_edges"]
        }

        if result["exact_match"]:
            log_experiment_csv(csv_log_entry, log_file_names["successful"])
        else:
            log_experiment_csv(csv_log_entry, log_file_names["failed"])

        return result
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None


def run_multiple_experiments(client: OpenAI, df: DataFrame, num_experiments: int, log_file_names: dict) -> None:
    """
    Run multiple experiments and calculate aggregate metrics.

    :param log_file_names: Dictionary containing the filenames for successful and failed experiments.
    :param client: OpenAI API client.
    :param df: DataFrame containing the questions and expected answers.
    :param num_experiments: Number of experiments to run.
    """
    results = []
    failed_experiments = 0

    for _ in tqdm(range(num_experiments), desc="Running Experiments"):
        result = run_single_experiment(
            client=client,
            df=df,
            log_file_names=log_file_names
        )
        if result:
            results.append(result)
        else:
            failed_experiments += 1

    # Aggregate metrics from multiple experiments
    if results:
        aggregated_metrics = aggregate_metrics(results)
        display_metrics(aggregated_metrics)

    print(f"\nTotal failed experiments: {failed_experiments} out of {num_experiments}")
    print(f"\nTotal failed experiments: {failed_experiments} out of {num_experiments}")
    print(f"Results logged to:")
    print(f"  - Successful experiments: {log_file_names['successful']}")
    print(f"  - Failed experiments: {log_file_names['failed']}")


def main():
    ### CONFIG ###
    TEMPERATURE: int = 1
    DO_SAMPLE: bool = False
    NUM_EXPERIMENTS: int = 10
    NO_VARIABLES: int = 5

    log_files = get_log_filenames(
        TEMPERATURE,
        DO_SAMPLE,
        NUM_EXPERIMENTS,
        NO_VARIABLES
    )

    # Retrieve the API key from the .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")

    # Load the dataframe
    csv_file_path = "data/v0.0.7/train.csv"
    df = pd.read_csv(csv_file_path)

    # Initialize the API client
    client = OpenAI(
        api_key=api_key,
    )

    # Run multiple experiments
    run_multiple_experiments(
        client=client,
        df=df,
        num_experiments=NUM_EXPERIMENTS,
        log_file_names=log_files
    )


if __name__ == "__main__":
    main()