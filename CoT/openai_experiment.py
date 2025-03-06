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
- Extract and list ALL marginal and conditional independence statements.

Step 2: Initialize the Graph
- Create edges between all correlated node pairs.
- List connections for each node.

Step 3: Systematic Independence Testing
- For EACH edge in the initial graph:
  a. Test against EACH independence statement INDIVIDUALLY
  b. Document your reasoning for each test
  c. Decide whether to keep or remove the edge
- Show your work using EXACTLY this format:
  * Edge X -- Y:
    - Testing against statement 1: [exact statement] → [reasoning] → [consistent/inconsistent]
    - Testing against statement 2: [exact statement] → [reasoning] → [consistent/inconsistent]
    - [Continue for ALL independence statements]
    - Decision: [keep/remove] edge X -- Y

Step 4: Compile the Causal Undirected Skeleton
- Construct the final graph structure
- List each node with its connected nodes.
- Ensure all applied independencies are reflected accurately.

CRITICAL REMINDER ABOUT d-SEPARATION:
- An edge X -- Y should be REMOVED if and only if there exists ANY independence statement indicating that X and Y are independent (either marginally or conditional on some set of variables).
- The absence of an independence statement means X and Y are dependent and the edge should be KEPT.
- Do not assume independencies that are not explicitly stated in the premise

IMPORTANT: The output format of Step 4 must match EXACTLY the following example:
Step 4: Compile the Causal Undirected Skeleton
In this graph:
  - Node A is connected to nodes B, C.
  - Node B is connected to nodes A, C.
  - Node C is connected to nodes A, B.
"""

user_prompt = """
Example:

Task Description: Given the provided premise, apply the PC (Peter-Clark) algorithm following the described step-by-step instructions to compute the causal undirected skeleton.
Premise: Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: A correlates with C. A correlates with D. A correlates with E. B correlates with C. B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. However, A is independent of B. A and D are independent given B and C. A and D are independent given C. B and D are independent given A and C. B and D are independent given C.
Answer:
Step 1: Read the Data
- Extracted nodes: A, B, C, D, E
- Correlations:
  - Node A is correlated with nodes C, D, E.
  - Node B is correlated with nodes C, D, E.
  - Node C is correlated with nodes A, B, D, E.
  - Node D is correlated with nodes A, B, C, E.
  - Node E is correlated with nodes A, B, C, D.
- Independence Statements:
  1. A is independent of B
  2. A and D are independent given B and C
  3. A and D are independent given C 
  4. B and D are independent given A and C
  5. B and D are independent given C

Step 2: Initialize the Graph
Created edges between all correlated variable pairs. In this graph:
  - Node A is connected to nodes C, D, E.
  - Node B is connected to nodes C, D, E.
  - Node C is connected to nodes A, B, D, E.
  - Node D is connected to nodes A, B, C, E.
  - Node E is connected to nodes A, B, C, D.

Step 3: Systematic Independence Testing
* Edge A -- C:
  - Statement 1: A is independent of B → This is about A-B independence, not A-C → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not A-C → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not A-C → not relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not A-C → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not A-C → not relevant
  - Decision: keep edge A -- C based on no relevant statements

* Edge A -- D:
  - Statement 1: A is independent of B → This is about A-B independence, not A-D → not relevant
  - Statement 2: A and D are independent given B and C → This directly states A and D are independent when conditioned on B and C → relevant
  - Statement 3: A and D are independent given C → This directly states A and D are independent when conditioned on C → relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not A-D → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not A-D → not relevant
  - Decision: remove edge A -- D based on statements 2 and 3

* Edge A -- E:
  - Statement 1: A is independent of B → This is about A-B independence, not A-E → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not A-E → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not A-E → not relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not A-E → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not A-E → not relevant
  - Decision: keep edge A -- E based on no relevant statements

* Edge B -- C:
  - Statement 1: A is independent of B → This is about A-B independence, not B-C → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not B-C → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not B-C → not relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not B-C → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not B-C → not relevant
  - Decision: keep edge B -- C based on no relevant statements

* Edge B -- D:
  - Statement 1: A is independent of B → This is about A-B independence, not B-D → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not B-D → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not B-D → not relevant
  - Statement 4: B and D are independent given A and C → This directly states B and D are independent when conditioned on A and C → relevant
  - Statement 5: B and D are independent given C → This directly states B and D are independent when conditioned on C → relevant
  - Decision: remove edge B -- D based on statements 4 and 5

* Edge B -- E:
  - Statement 1: A is independent of B → This is about A-B independence, not B-E → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not B-E → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not B-E → not relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not B-E → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not B-E → not relevant
  - Decision: keep edge B -- E based on no relevant statements

* Edge C -- D:
  - Statement 1: A is independent of B → This is about A-B independence, not C-D → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not C-D → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not C-D → not relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not C-D → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not C-D → not relevant
  - Decision: keep edge C -- D based on no relevant statements

* Edge C -- E:
  - Statement 1: A is independent of B → This is about A-B independence, not C-E → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not C-E → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not C-E → not relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not C-E → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not C-E → not relevant
  - Decision: keep edge C -- E based on no relevant statements

* Edge D -- E:
  - Statement 1: A is independent of B → This is about A-B independence, not D-E → not relevant
  - Statement 2: A and D are independent given B and C → This is about A-D independence, not D-E → not relevant
  - Statement 3: A and D are independent given C → This is about A-D independence, not D-E → not relevant
  - Statement 4: B and D are independent given A and C → This is about B-D independence, not D-E → not relevant
  - Statement 5: B and D are independent given C → This is about B-D independence, not D-E → not relevant
  - Decision: keep edge D -- E based on no relevant statements

Step 4: Compile the Causal Undirected Skeleton
In this graph:
  - Node A is connected to nodes C, E.
  - Node B is connected to nodes C, E.
  - Node C is connected to nodes A, B, D, E.
  - Node D is connected to nodes C, E.
  - Node E is connected to nodes A, B, C, D.

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
LOGS_DIR = "logs"


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
        prompt_data = generate_few_shot_prompt(df, num_examples=0)
        prompt_content = "\n".join(prompt_data["standard_prompt"])

        # Debug: Print the system prompt
        print("\nSystem Prompt:")
        print(system_prompt)

        # Debug: Print the generated prompt
        print("\nGenerated Prompt:")
        print(prompt_content)

        new_prompt = system_prompt + "\n" + prompt_content
        print("hehe new prompt\n")
        print(new_prompt)

        # Expected answer
        new_question_index = prompt_data["new_question_index"]
        question_row = df.iloc[new_question_index]
        expected_answer = question_row["expected_answer"]

        # Extract edges from the expected answer
        expected_edges = extract_edges_incident_format(answer=expected_answer, step=5)

        # Generate the answer using the OpenAI API
        completion = client.chat.completions.create(
            # model="gpt-4o-mini",
            model="o1-mini",
            # reasoning_effort="medium",
            messages=[
                # {
                #     "role": "developer",
                #     "content": system_prompt
                # },
                {
                    "role": "user",
                    "content": new_prompt
                }
            ]
        )

        model_response = completion.choices[0].message.content

        # Debug: Print the model's response
        print("\nModel Response:")
        print(model_response)

        # Extract edges from the model's answer
        answer_edges = extract_edges_incident_format(answer=model_response, step=4)

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
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, "data/v0.0.6/train.csv")
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