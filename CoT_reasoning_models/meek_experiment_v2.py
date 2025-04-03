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

from CoT_reasoning_models.meek_rules import apply_meek_rules
from answer_extractor import extract_edges_incident_format, extract_premise, extract_separation_sets, find_v_structures, \
    aggregate_metrics, display_metrics, extract_directed_edges_json, compare_directed_edges, \
    extract_directed_edges_literal_format_json

CSV_FIELDS_MEEK: list[str] = [
    "sample_id",
    "raw_input",
    "raw_label",
    "raw_template",
    "skeleton_expected_answer",
    "skeleton_edges",
    "v_structures",
    "input_prompt",
    "model_answer",
    "expected_directed_edges",
    "model_directed_edges",
    "missing_edges",
    "extra_edges",
    "reversed_edges"
]
LOGS_DIR: str = "logs"


def ensure_logs_directory_exists():
    """
    Create the logs directory if it doesn't exist.
    """
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        print(f"Created logs directory: {LOGS_DIR}")


def log_meek_experiment_csv(entry: dict, file_path: str) -> None:
    """
    Log a Meek rules experiment to the appropriate CSV file.

    :param entry: Dictionary containing the experiment data.
    :param file_path: Path to the CSV log file.
    """
    # Convert sets/lists to string representations for CSV
    entry["expected_directed_edges"] = str(list(entry["expected_directed_edges"]))
    entry["model_directed_edges"] = str(list(entry["model_directed_edges"]))
    entry["missing_edges"] = str(list(entry["missing_edges"]))
    entry["extra_edges"] = str(list(entry["extra_edges"]))
    entry["reversed_edges"] = str(list(entry["reversed_edges"]))

    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS_MEEK)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


def get_meek_log_filenames(temperature: int, do_sample: bool, num_experiments: int, no_variables: int) -> dict:
    """
    Generate log filenames for Meek rules experiments based on configuration parameters.
    """
    # Ensure logs directory exists
    ensure_logs_directory_exists()

    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%d-%b-%Y-%H%M")

    # Build the base filename
    base_filename = f"{date_str}-meek-temp{temperature}"

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


def run_single_experiment(client: OpenAI, row: pd.Series, idx: int, log_file_names: dict) -> Optional[dict]:
    num_variables: int = row["num_variables"]
    if num_variables == 2:
        nodes = ["A", "B"]
    elif num_variables == 3:
        nodes = ["A", "B", "C"]
    elif num_variables == 4:
        nodes = ["A", "B", "C", "D"]
    elif num_variables == 5:
        nodes = ["A", "B", "C", "D", "E"]
    elif num_variables == 6:
        nodes = ["A", "B", "C", "D", "E", "F"]
    else:
        raise ValueError("Number of variables must between 2 and 6.")

    try:
        premise = extract_premise(row["input"])

        # Undirected skeleton graph edges
        sample_answer = row["expected_answer"]
        graph_edges = extract_edges_incident_format(answer=sample_answer, step=5)

        # Extract the expected V-structures
        separation_sets = extract_separation_sets(premise=premise)
        expected_v_structures = find_v_structures(skeleton_edges=graph_edges, separation_sets=separation_sets)

        # Format the edges with line breaks
        formatted_edges = "[\n    "
        formatted_edges += ",\n    ".join([str(edge) for edge in graph_edges])
        formatted_edges += "\n  ]"

        # Prepare the prompt
        prompt_template = PROMPTS["reasoning_prompt_meek_v4"]
        prompt = prompt_template.format(premise=premise,
                                        v_structures=expected_v_structures,
                                        nodes=nodes,
                                        edges=formatted_edges)
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

        # Extract the expected directed edges
        expected_directed_edges = apply_meek_rules(skeleton_edges=graph_edges, v_structures=expected_v_structures)
        print(f"\nExpected directed edges:\n{expected_directed_edges}")

        # Extract the directed edges from the model response
        answer_directed_edges = extract_directed_edges_literal_format_json(answer=model_response)
        print(f"\nModel directed edges:\n{answer_directed_edges}")

        result = compare_directed_edges(expected_edges=expected_directed_edges, model_edges=answer_directed_edges)
        print(f"\nComparison result:\n{result}")

        # Log the results
        csv_log_entry = {
            "sample_id": idx,
            "raw_input": row["input"],
            "raw_label": row["label"],
            "raw_template": row["template"],
            "skeleton_expected_answer": sample_answer,
            "skeleton_edges": graph_edges,
            "v_structures": expected_v_structures,
            "input_prompt": prompt,
            "model_answer": model_response,
            "expected_directed_edges": expected_directed_edges,
            "model_directed_edges": answer_directed_edges,
            "missing_edges": result["missing_edges"],
            "extra_edges": result["extra_edges"],
            "reversed_edges": result["reversed_edges"]
        }

        if result["exact_match"]:
            log_meek_experiment_csv(csv_log_entry, log_file_names["successful"])
        else:
            log_meek_experiment_csv(csv_log_entry, log_file_names["failed"])

        return result
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None


def run_multiple_experiments(client: OpenAI, df: DataFrame, num_experiments: int, log_file_names: dict) -> None:
    """
    Run multiple  experiments and calculate aggregate metrics.

    :param client: OpenAI API client.
    :param df: DataFrame containing the questions and expected answers.
    :param num_experiments: Number of experiments to run.
    :param log_file_names: Dictionary containing the filenames for successful and failed experiments.
    """
    results = []
    failed_experiments = 0
    failed_ids = []

    debug_flag = False
    debug_index = 2168
    if debug_flag:
        sampled_rows = df.loc[[debug_index]]
        num_samples = len(sampled_rows)
        print(f"Processing specific row with index {debug_index}")
    else:
        num_samples = min(num_experiments, len(df))
        sampled_rows = df.sample(n=num_samples, replace=False)
        print(f"Selected row indices: {list(sampled_rows.index)}")

    for idx, row in tqdm(sampled_rows.iterrows(), total=num_samples, desc="Running Experiments"):
        print(f"\nProcessing row with index {idx}")
        result = run_single_experiment(
            client=client,
            row=row,
            idx=idx,
            log_file_names=log_file_names
        )
        if result:
            results.append(result)
        else:
            failed_experiments += 1
            failed_ids.append(idx)

    # Aggregate metrics from multiple experiments
    if results:
        aggregated_metrics = aggregate_metrics(results)
        display_metrics(aggregated_metrics)

    print(f"Results logged to:")
    print(f"  - Successful experiments: {log_file_names['successful']}")
    print(f"  - Failed experiments: {log_file_names['failed']}")

    print(f"\nTotal failed experiments: {failed_experiments} out of {num_experiments}")
    if failed_ids:
        print(f"Failed experiment IDs: {failed_ids}")


def main():
    ### CONFIG ###
    TEMPERATURE: int = 1
    DO_SAMPLE: bool = False
    NUM_EXPERIMENTS: int = 20
    NO_VARIABLES: int = 5

    log_files = get_meek_log_filenames(
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
