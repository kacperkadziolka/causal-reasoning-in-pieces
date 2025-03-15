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

from answer_extractor import extract_edges_incident_format, extract_premise, extract_separation_sets, find_v_structures, \
    extract_vstructures_json, compare_vstructures, aggregate_metrics, display_metrics

CSV_FIELDS_VSTRUCTURES: list[str] = [
    "input_prompt",
    "expected_answer",
    "model_answer",
    "expected_vstructs",
    "model_vstructs",
    "missing_vstructs",
    "extra_vstructs"
]
LOGS_DIR: str = "logs"


def ensure_logs_directory_exists():
    """
    Create the logs directory if it doesn't exist.
    """
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        print(f"Created logs directory: {LOGS_DIR}")


def log_vstructures_experiment_csv(entry: dict, file_path: str) -> None:
    """
    Log a v-structures experiment to the appropriate CSV file.

    :param entry: Dictionary containing the experiment data.
    :param file_path: Path to the CSV log file.
    """
    # Convert sets/lists to string representations for CSV
    entry["expected_vstructs"] = str(list(entry["expected_vstructs"]))
    entry["model_vstructs"] = str(list(entry["model_vstructs"]))
    entry["missing_vstructs"] = str(list(entry["missing_vstructs"]))
    entry["extra_vstructs"] = str(list(entry["extra_vstructs"]))

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS_VSTRUCTURES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


def get_vstructure_log_filenames(temperature: int, do_sample: bool, num_experiments: int, no_variables: int) -> dict:
    """
    Generate log filenames for v-structure experiments based on configuration parameters.
    """
    # Ensure logs directory exists
    ensure_logs_directory_exists()

    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%d-%b-%Y-%H%M")

    # Build the base filename
    base_filename = f"{date_str}-vstructs-temp{temperature}"

    if do_sample:
        base_filename += "-dosample"

    base_filename += f"-{num_experiments}exp-{no_variables}var"

    return {
        "successful": os.path.join(LOGS_DIR, f"{base_filename}-successful.csv"),
        "failed": os.path.join(LOGS_DIR, f"{base_filename}-failed.csv")
    }


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


def run_single_experiment(client: OpenAI, df: DataFrame, log_file_names: dict) -> Optional[dict]:
    """
    Run a single experiment to compare the expected V-structures with the model's predicted V-structures.
    """
    nodes = ["A", "B", "C", "D", "E"]
    try:
        # Draw a sample
        sample = df.sample(n=1)
        premise = extract_premise(sample["input"].iloc[0])
        print(f"\nSelected a sample No. {sample.index[0]}.")

        # Undirected skeleton graph edges
        sample_answer = sample["expected_answer"].iloc[0]
        graph_edges = extract_edges_incident_format(answer=sample_answer, step=5)

        # Extract the expected V-structures
        separation_sets = extract_separation_sets(premise=premise)
        expected_v_structures = find_v_structures(skeleton_edges=graph_edges, separation_sets=separation_sets)
        print(f"Expected V-structures: {expected_v_structures}")

        # Format the edges with line breaks
        formatted_edges = "[\n    "
        formatted_edges += ",\n    ".join([str(edge) for edge in graph_edges])
        formatted_edges += "\n  ]"

        # Prepare the prompt
        prompt_template = PROMPTS["reasoning_prompt_vstructure"]
        prompt = prompt_template.format(premise=premise, nodes=nodes, edges=formatted_edges)
        print(f"\nInput prompt:\n{prompt}")

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

        # Extract the V-structures from the model response
        answer_v_structures = extract_vstructures_json(answer=model_response)
        print(f"\nModel V-structures:\n{answer_v_structures}")

        result = compare_vstructures(expected_vstructs=expected_v_structures, answer_vstructs=answer_v_structures)
        print(f"\nComparison result:\n{result}")

        # Log the results
        csv_log_entry = {
            "input_prompt": prompt,
            "expected_answer": sample_answer,
            "model_answer": model_response,
            "expected_vstructs": expected_v_structures,
            "model_vstructs": answer_v_structures,
            "missing_vstructs": result["missing_vstructs"],
            "extra_vstructs": result["extra_vstructs"]
        }

        if result["exact_match"]:
            log_vstructures_experiment_csv(csv_log_entry, log_file_names["successful"])
        else:
            log_vstructures_experiment_csv(csv_log_entry, log_file_names["failed"])

        return result
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None


def run_multiple_experiments(client: OpenAI, df: DataFrame, num_experiments: int, log_file_names: dict) -> None:
    """
    Run multiple v-structure experiments and calculate aggregate metrics.

    :param client: OpenAI API client.
    :param df: DataFrame containing the questions and expected answers.
    :param num_experiments: Number of experiments to run.
    :param log_file_names: Dictionary containing the filenames for successful and failed experiments.
    """
    results = []
    failed_experiments = 0

    for _ in tqdm(range(num_experiments), desc="Running V-Structure Experiments"):
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
    print(f"Results logged to:")
    print(f"  - Successful experiments: {log_file_names['successful']}")
    print(f"  - Failed experiments: {log_file_names['failed']}")


def main():
    ### CONFIG ###
    TEMPERATURE: int = 1
    DO_SAMPLE: bool = False
    NUM_EXPERIMENTS: int = 1
    NO_VARIABLES: int = 5

    log_files = get_vstructure_log_filenames(
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
    csv_file_path = "data/v0.0.6/train.csv"
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
