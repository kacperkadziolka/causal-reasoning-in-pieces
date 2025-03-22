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


def run_single_experiment(client: OpenAI, row: pd.Series) -> Optional[dict]:
    try:
        # Draw sample
        premise = extract_premise(row["input"])
        hypothesis = extract_hypothesis(row["input"])

        # Prepare the prompt
        prompt_template = PROMPTS["reasoning_prompt_single_stage"]
        prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)
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

    num_samples = min(num_experiments, len(df))
    sampled_rows = df.sample(n=num_samples, replace=False)

    for _, row in tqdm(sampled_rows.iterrows(), total=num_samples, desc="Running Experiments"):
        result = run_single_experiment(
            client=client,
            row=row,
        )
        if result:
            results.append(result)
        else:
            failed_experiments += 1

    if results:
        aggregated_metrics = aggregate_metrics_single_prompt(results)
        display_metrics(aggregated_metrics)


def main():
    NUM_EXPERIMENTS: int = 10

    # Retrieve the API key from the .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")

    # Load the dataframe
    csv_file_path = "data/test/balanced_50_50_test.csv"
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
    )


if __name__ == "__main__":
    main()