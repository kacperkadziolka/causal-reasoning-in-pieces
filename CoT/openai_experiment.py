import csv
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pandas import DataFrame
from tqdm import tqdm

from CoT.answer_extractor import extract_edges_incident_format, compare_edges, aggregate_metrics, display_metrics
from CoT.prompt_generator import generate_few_shot_prompt


system_prompt = """
You are an expert in causal inference and data analysis, proficient in applying the PC (Peter-Clark) algorithm. 
Follow these steps in the provided order to respond accurately:

Step 1: Read the Data
- Identify extracted nodes and their correlations.
- Note marginal and conditional independencies.

Step 2: Initialize the Graph
- Create edges between all correlated node pairs.
- List connections for each node.

Step 3: Apply Marginal Independencies
- Remove edges based on marginal independencies.
- Specify removed edges, if any.

Step 4: Apply Conditional Independencies
- Remove edges based on conditional independencies.
- Specify which independencies led to each removal.

Step 5: Compile the Causal Undirected Skeleton
- Construct the final graph structure
- List each node with its connected nodes.
- Ensure all applied independencies are reflected accurately.

**Example of Step 5:**

Step 5: Compile the Causal Undirected Skeleton
In this graph:
  - Node A is connected to nodes B, C, D.
  - Node B is connected to nodes A, C.
  - Node C is connected to nodes A, B.
  - Node D is connected to node A.
"""


CSV_FIELDS = [
    "input_prompt",
    "expected_answer",
    "model_answer",
    "expected_edges",
    "model_edges",
    "missing_edges",
    "extra_edges"
]

def get_log_filenames(temperature: int, do_sample: bool, num_experiments: int, no_variables: int) -> dict:
    """
    Generate log filenames based on configuration parameters.

    :param temperature: Temperature value used for generation
    :param do_sample: Whether sampling was used
    :param num_experiments: Number of experiments
    :param no_variables: Number of variables in the experiment
    :return: Dictionary with filenames for successful and failed experiments
    """
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%d-%b-%Y-%H%M")

    # Build the base filename
    base_filename = f"{date_str}-temp{temperature}"

    if do_sample:
        base_filename += "-dosample"

    base_filename += f"-{num_experiments}exp-{no_variables}var"

    return {
        "successful": f"{base_filename}-successful.csv",
        "failed": f"{base_filename}-failed.csv"
    }


def log_experiment_csv(entry: dict, file_path: str) -> None:
    """
    Log an experiment to the appropriate CSV file.

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
        # Generate the prompt for LLM
        print("\nGenerating a few-shot prompt...")
        prompt_data = generate_few_shot_prompt(df, num_examples=3)
        prompt_content = "\n".join(prompt_data["standard_prompt"])

        # Debug: Print the system prompt
        print("\nSystem Prompt:")
        print(system_prompt)

        # Debug: Print the generated prompt
        print("\nGenerated Prompt:")
        print(prompt_content)

        # Expected answer
        new_question_index = prompt_data["new_question_index"]
        question_row = df.iloc[new_question_index]
        expected_answer = question_row["expected_answer"]

        # Extract edges from the expected answer
        expected_edges = extract_edges_incident_format(expected_answer)

        # Generate the answer using the OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        )

        model_response = completion.choices[0].message.content

        # Debug: Print the model's response
        print("\nModel Response:")
        print(model_response)

        # Extract edges from the model's answer
        answer_edges = extract_edges_incident_format(model_response)

        # Compare the expected edges with the model's predicted edges
        result = compare_edges(expected_edges, answer_edges)

        csv_log_entry = {
            "input_prompt": prompt_content,
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
        return None  # Skip the experiment if an error occurs


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
    NUM_EXPERIMENTS: int = 5
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
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, "data/v0.0.6/train.csv")
    df = pd.read_csv(csv_file_path)

    # Initialize the API client
    client = OpenAI(
        api_key=api_key,
    )

    # Run a single experiment
    # print(run_single_experiment(client, df))

    # Run multiple experiments
    run_multiple_experiments(
        client=client,
        df=df,
        num_experiments=NUM_EXPERIMENTS,
        log_file_names=log_files
    )


if __name__ == "__main__":
    main()