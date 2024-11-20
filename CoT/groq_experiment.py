import ast
import os
import re
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pandas import DataFrame
from tqdm import tqdm

from prompt_generator import generate_few_shot_prompt


def extract_edges(answer: str) -> set:
    """
    Extract the causal edges from the provided answer string.

    :param answer: The expected answer or answer returned by the GROQ API.
    :return: A set of edges extracted from the answer.
    """
    try:
        step_5_index = answer.find("Step 5: Compile the Causal Undirected Skeleton")
        if step_5_index == -1:
            raise ValueError("Step 5 section not found in the answer.")

        edges_start = answer.find("Edges:", step_5_index) + len("Edges:")
        edges_raw = answer[edges_start:].strip()

        match = re.search(r"{(.*?)}", edges_raw)
        if not match:
            raise ValueError("Edges section not properly formatted or missing braces.")

        # Extract the content inside the braces and add quotes around edge variables (e.g., (E,  B) -> ('E', 'B'))
        edges_content = match.group(0)
        edges_content = re.sub(r"(\w+)", r"'\1'", edges_content)

        # Parse the edges and normalize them as sorted tuples
        edges = set(tuple(sorted(edge)) for edge in ast.literal_eval(edges_content))
        return edges
    except Exception as e:
        raise RuntimeError(f"Failed to extract edges: {e}")


def extract_edges_incident_format(answer: str) -> set:
    """
    Extract the causal edges from the provided answer string using the adjacency list format.

    :param answer: The expected answer or answer returned by the GROQ API.
    :return: A set of edges extracted from the answer, represented as sorted tuples.
    """
    #print("Answer:", answer)
    try:
        # Locate Step 5
        step_5_pattern = r"Step 5: Compile the Causal Undirected Skeleton"
        step_5_match = re.search(step_5_pattern, answer)
        if not step_5_match:
            raise ValueError("Step 5 section not found in the answer.")

        # Find the start of the adjacency list
        adjacency_start = answer.find("In this graph:", step_5_match.end())
        if adjacency_start == -1:
            raise ValueError("Adjacency list section not found in Step 5.")

        # Extract the adjacency list lines
        adjacency_section = answer[adjacency_start:].splitlines()
        #print("Adjacency Section:", adjacency_section)

        # Initialize set for edges
        edges = set()

        # Iterate over each line in the adjacency section
        for line in adjacency_section:
            line = line.strip()
            #print("Line:", line)
            # Match lines like: "- Node A is connected to nodes C, D, F."
            node_conn_match = re.match(r"- Node (\w+) is connected to nodes (.+)\.", line)
            if node_conn_match:
                node = node_conn_match.group(1)
                connections = node_conn_match.group(2).split(", ")
                #print(f"Node: {node}, Connections: {connections}")
                for conn in connections:
                    # Add edge as sorted tuple to avoid duplicates
                    edge = tuple(sorted([node, conn]))
                    #print("Edge:", edge)
                    edges.add(edge)
                continue  # Proceed to next line after processing
        #print("Edges:", edges)
        return edges
    except Exception as e:
        raise RuntimeError(f"Failed to extract edges: {e}")


def compare_edges(expected_edges: set, answer_edges: set) -> dict:
    """
    Compare the expected edges with the model's predicted edges.

    :param expected_edges: The ground-truth edges.
    :param answer_edges: The model's predicted edges.
    :return: A dictionary with metrics for comparison.
    """
    # Check if the edges are an exact match
    exact_match = expected_edges == answer_edges

    if exact_match:
        print("The model's predicted edges match the expected edges!")
    else:
        print("Mismatch found:")
        print(f"Expected edges: {expected_edges}")
        print(f"Model edges: {answer_edges}")
        print(f"Missing in model: {expected_edges - answer_edges}")
        print(f"Extra in model: {answer_edges - expected_edges}")

    # Calculate detailed metrics
    true_positive = len(expected_edges & answer_edges)  # Correctly predicted edges
    false_positive = len(answer_edges - expected_edges)  # Extra edges in prediction
    false_negative = len(expected_edges - answer_edges)  # Missing edges in prediction

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "expected_count": len(expected_edges),
        "predicted_count": len(answer_edges),
        "exact_match": exact_match,
    }


def aggregate_metrics(results: list) -> dict:
    """
    Aggregate metrics from multiple experiments.

    :param results: A list of comparison results from multiple experiments.
    :return: A dictionary with aggregated metrics.
    """
    total_true_positive = sum(r["true_positive"] for r in results)
    total_false_positive = sum(r["false_positive"] for r in results)
    total_false_negative = sum(r["false_negative"] for r in results)
    total_expected_count = sum(r["expected_count"] for r in results)
    total_exact_matches = sum(1 for r in results if r["exact_match"])
    total_prompts = len(results)

    precision = total_true_positive / (total_true_positive + total_false_positive) if (total_true_positive + total_false_positive) > 0 else 0
    recall = total_true_positive / (total_true_positive + total_false_negative) if (total_true_positive + total_false_negative) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_true_positive / total_expected_count if total_expected_count > 0 else 0
    exact_match_ratio = total_exact_matches / total_prompts if total_prompts > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "exact_match_ratio": exact_match_ratio,
        "total_exact_matches": total_exact_matches,
        "total_prompts": total_prompts,
    }


def display_metrics(metrics: dict) -> None:
    """
    Display all the metrics in a user-friendly format.

    :param metrics: Dictionary of aggregated metrics.
    """
    print("\nAggregate Metrics:")
    for key, value in metrics.items():
        # Format floating-point numbers with 2 decimal places
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').capitalize()}: {value}")


def run_single_experiment(client: Groq, df: DataFrame) -> Optional[dict]:
    """
    Run a single experiment to compare expected edges with the model's predicted edges.

    :param client: GROQ API client.
    :param df: DataFrame containing the questions and expected answers.
    :return: A dictionary with the comparison result.
    """
    try:
        # Generate the prompt for LLM
        print("\nGenerating a few-shot prompt...")
        prompt_data = generate_few_shot_prompt(df, num_examples=3)
        prompt_content = "\n".join(prompt_data["standard_prompt"])

        # Expected answer
        new_question_index = prompt_data["new_question_index"]
        question_row = df.iloc[new_question_index]
        expected_answer = question_row["expected_answer"]

        # Extract edges from the expected answer
        expected_edges = extract_edges_incident_format(expected_answer)

        # Generate the answer using the GROQ API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            model="llama3-70b-8192",
        )

        # Debug: Print the model's response
        print("\nModel Response:")
        print(chat_completion.choices[0].message.content)

        # Extract edges from the model's answer
        answer_edges = extract_edges_incident_format(chat_completion.choices[0].message.content)

        # Compare the expected edges with the model's predicted edges
        return compare_edges(expected_edges, answer_edges)
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None  # Skip the experiment if an error occurs


def run_multiple_experiments(client: Groq, df: DataFrame, num_experiments: int) -> None:
    """
    Run multiple experiments and calculate aggregate metrics.

    :param client: GROQ API client.
    :param df: DataFrame containing the questions and expected answers.
    :param num_experiments: Number of experiments to run.
    """
    results = []
    failed_experiments = 0

    for _ in tqdm(range(num_experiments), desc="Running Experiments"):
        result = run_single_experiment(client, df)
        if result:
            results.append(result)
        else:
            failed_experiments += 1

        # Throttle the requests to avoid Groq rate limiting
        print("Throttling: Waiting for 1 minute and 5 seconds before the next request...")
        time.sleep(65)

    # Aggregate metrics from multiple experiments
    if results:
        aggregated_metrics = aggregate_metrics(results)
        display_metrics(aggregated_metrics)

    print(f"\nTotal failed experiments: {failed_experiments} out of {num_experiments}")


def main():
    # Retrieve the API key from the .env file
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')

    if not api_key:
        raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable in your .env file.")

    # Load the dataframe
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, "data/v0.0.3/train.csv")
    df = pd.read_csv(csv_file_path)

    # Initialize the GROQ client
    client = Groq(
        api_key=api_key,
    )

    # Run a single experiment
    #print(run_single_experiment(client, df))

    # Run multiple experiments
    run_multiple_experiments(client, df, num_experiments=50)


if __name__ == "__main__":
    main()
