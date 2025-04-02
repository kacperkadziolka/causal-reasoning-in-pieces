import os
import pandas as pd
import yaml
from pandas import Series

from answer_extractor import extract_premise, extract_hypothesis


def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

PROMPTS = load_yaml("prompts.yaml")


def prepare_experiment_from_row(row: Series) -> dict:
    """
    Create an experiment dictionary from a DataFrame row.
    """
    sample_id = row.name
    premise = extract_premise(row["input"])
    hypothesis = extract_hypothesis(row["input"])
    prompt = PROMPTS["single_stage_prompt"].format(premise=premise, hypothesis=hypothesis)

    return {
        "input": row["input"],
        "label": row["label"],
        "num_variables": row["num_variables"],
        "template": row["template"],
        "sampleId": sample_id,
        "prompt": prompt,
        "premise": premise,
        "hypothesis": hypothesis,
        "model_answer": None,
        "model_label": None,
        "attempt_count": 0,
    }

def append_log(log_file: str, log_entry: dict) -> None:
    """
    Append a single log entry to a CSV file.
    """
    log_df = pd.DataFrame([log_entry])

    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False, mode='w')
    else:
        log_df.to_csv(log_file, index=False, header=False, mode='a')
