import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pandas import DataFrame
from tqdm import tqdm

from CoT.answer_extractor import extract_edges_incident_format
from prompt_generator import generate_few_shot_prompt


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
        prompt_data = generate_few_shot_prompt(df, num_examples=2)
        prompt_content = "\n".join(prompt_data["standard_prompt"])

        # Debug: Print the system prompt
        # print("\nSystem Prompt:")
        # print(system_prompt)

        # Debug: Print the generated prompt
        # print("\nGenerated Prompt:")
        # print(prompt_content)

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
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            model="llama-3.3-70b-versatile",
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
        print("Throttling: Waiting for 1.5 minute before the next request...")
        time.sleep(90)

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
    csv_file_path = os.path.join(script_directory, "data/v0.0.6/train.csv")
    df = pd.read_csv(csv_file_path)

    # Initialize the GROQ client
    client = Groq(
        api_key=api_key,
    )

    # Run a single experiment
    # print(run_single_experiment(client, df))

    # Run multiple experiments
    run_multiple_experiments(client, df, num_experiments=50)


if __name__ == "__main__":
    main()
