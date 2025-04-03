import os
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pandas import DataFrame
from tqdm import tqdm

from CoT_reasoning_models.meek_rules import apply_meek_rules
from answer_extractor import extract_premise, extract_hypothesis, extract_hypothesis_answer, compare_hypothesis_answers, \
    aggregate_metrics_single_prompt, display_metrics, extract_edges_incident_format, extract_separation_sets, \
    find_v_structures


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


def run_single_experiment(client: OpenAI, row: pd.Series, debug_flag: bool) -> Optional[dict]:
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
        # Extract hypothesis
        premise = extract_premise(row["input"])
        hypothesis = extract_hypothesis(row["input"])

        # Extract directed edges
        sample_answer = row["expected_answer"]
        graph_edges = extract_edges_incident_format(answer=sample_answer, step=5)
        separation_sets = extract_separation_sets(premise=premise)
        expected_v_structures = find_v_structures(skeleton_edges=graph_edges, separation_sets=separation_sets)
        expected_directed_edges = apply_meek_rules(skeleton_edges=graph_edges, v_structures=expected_v_structures)

        print(f"\nExpected directed edges:\n{expected_directed_edges}")
        print(f"\nGraph edges:\n{graph_edges}")

        # Compute the undirected edges
        # These are all the causal edges from the skeleton that remains undirected
        undirected_edges = graph_edges - expected_directed_edges
        print("Undirected edges:", undirected_edges)

        if debug_flag:
            print(f"\nPremise:\n{premise}")
            print(f"\nHypothesis:\n{hypothesis}")
            print(f"\nUndirected skeleton edges:\n{graph_edges}")
            print(f"\nSeparation sets:\n{separation_sets}")
            print(f"\nV-structures:\n{expected_v_structures}")
            print(f"\nDirected skeleton edges:\n{expected_directed_edges}")

        # Format the edges with line breaks
        formatted_edges = "[\n    "
        formatted_edges += ",\n    ".join([str(edge) for edge in expected_directed_edges])
        formatted_edges += "\n  ]"

        formatted_edges_v2 = "[\n    "
        formatted_edges_v2 += ",\n    ".join([str(edge) for edge in undirected_edges])
        formatted_edges_v2 += "\n  ]"

        # Prepare the prompt
        prompt_template = PROMPTS["reasoning_prompt_apply_hypothesis"]
        prompt = prompt_template.format(premise=premise,
                                        nodes=nodes,
                                        directed_edges=formatted_edges,
                                        undirected_edges=formatted_edges_v2,
                                        hypothesis=hypothesis)
        print(f"\nInput prompt:\n{prompt}")

        # Ground truth label for given hypothesis
        label_int = row["label"]
        label_bool = bool(label_int)
        print(f"\nGround truth label:\n{label_bool}")

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
        answer_label = extract_hypothesis_answer(answer=model_response)
        print(f"\nModel label:\n{answer_label}")

        # Compare the expected edges with the model's predicted edges
        result = compare_hypothesis_answers(expected_answer=label_bool, model_answer=answer_label)
        print(f"\nComparison result:\n{result}")

        return result
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None


def run_multiple_experiments(client: OpenAI, df: DataFrame, num_experiments: int) -> None:
    results = []
    failed_experiments = 0
    failed_ids = []

    # Train set
    # Failed samples: 3060

    # Test set
    # Failed samples: 574, 705

    # Option to specify a single row to process (for debugging/testing)
    debug_flag = True
    debug_index = 705
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
            debug_flag=debug_flag
        )
        if result:
            results.append(result)
        else:
            failed_experiments += 1
            failed_ids.append(idx)

    if results:
        aggregated_metrics = aggregate_metrics_single_prompt(results)
        display_metrics(aggregated_metrics)

    # Print failure information
    print(f"\nTotal failed experiments: {failed_experiments}")
    if failed_ids:
        print(f"Failed experiment IDs: {failed_ids}")


def main():
    NUM_EXPERIMENTS: int = 30

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

    # Run multiple experiments
    run_multiple_experiments(
        client=client,
        df=df,
        num_experiments=NUM_EXPERIMENTS
    )


if __name__ == "__main__":
    main()