import json
import re


def extract_premise(text: str) -> str:
    """
    Extracts Premise from the input text.

    Parameters:
    - text (str): The input text containing "Premise:" and possibly "Hypothesis:".

    Returns:
    - str: The extracted premise.
    """
    # Use regex to extract text between "Premise:" and "Hypothesis:"
    match = re.search(r"Premise:\s*(.*?)\s*Hypothesis:", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If "Hypothesis:" is not present, extract everything after "Premise:"
        match = re.search(r"Premise:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # If "Premise:" is not found, return the original text
            return text.strip()


def extract_hypothesis(text: str) -> str:
    """
    Extracts Hypothesis from the input text.

    Parameters:
    - text (str): The input text containing "Hypothesis:".

    Returns:
    - str: The extracted hypothesis.
    """
    # Use regex to extract everything after "Hypothesis:"
    match = re.search(r"Hypothesis:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If "Hypothesis:" is not found, return empty string
        return ""


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


def extract_hypothesis_answer(answer: str) -> bool:
    """
    Extract the hypothesis answer (True/False) from the provided LLM answer string.

    Parameters:
    - answer (str): The answer returned by the LLM API.

    Returns:
    - bool: The extracted hypothesis answer as a Python boolean.

    Raises:
    - RuntimeError: If extraction fails.
    """
    try:
        # First approach: Try to parse the entire answer as JSON
        try:
            data = json.loads(answer)
            if "hypothesis_answer" in data:
                return bool(data["hypothesis_answer"])
        except json.JSONDecodeError:
            pass

        # Second approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*(.*?)```', answer, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                if "hypothesis_answer" in data:
                    return bool(data["hypothesis_answer"])
            except json.JSONDecodeError:
                pass

        # Third approach: Try to find complete JSON objects in the text
        potential_json_start = 0
        while potential_json_start < len(answer):
            start_idx = answer.find('{', potential_json_start)
            if start_idx == -1:
                break

            # Find matching closing brace by tracking brace balance
            open_braces = 1
            for end_idx in range(start_idx + 1, len(answer)):
                if answer[end_idx] == '{':
                    open_braces += 1
                elif answer[end_idx] == '}':
                    open_braces -= 1

                if open_braces == 0:
                    # Found a potential JSON object
                    try:
                        json_str = answer[start_idx:end_idx + 1]
                        data = json.loads(json_str)
                        if "hypothesis_answer" in data:
                            return bool(data["hypothesis_answer"])
                    except json.JSONDecodeError:
                        pass
                    break

            potential_json_start = start_idx + 1

        raise ValueError("No valid JSON with hypothesis_answer found")
    except Exception as e:
        raise RuntimeError(f"Failed to extract hypothesis answer: {e}")


def compare_hypothesis_answers(expected_answer: bool, model_answer: bool) -> dict:
    """
    Compare the expected hypothesis answer with the model's predicted answer.

    Parameters:
    - expected_answer (bool): The ground-truth answer (True/False).
    - model_answer (bool): The model's predicted answer (True/False).

    Returns:
    - dict: A dictionary with comparison metrics.
    """
    # Check if the answers match
    exact_match = expected_answer == model_answer

    # Calculate binary classification metrics
    true_positive = 1 if expected_answer and model_answer else 0
    true_negative = 1 if not expected_answer and not model_answer else 0
    false_positive = 1 if not expected_answer and model_answer else 0
    false_negative = 1 if expected_answer and not model_answer else 0

    if exact_match:
        print("\nThe model's predicted answer matches the expected answer!")
    else:
        print("\nMismatch found:")
        print(f"Expected answer: {expected_answer}")
        print(f"Model answer: {model_answer}")

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "exact_match": exact_match
    }


def aggregate_metrics_single_prompt(results: list) -> dict:
    """
    Aggregate metrics from multiple binary classification experiments.

    Parameters:
    - results (list): A list of comparison results from multiple experiments.

    Returns:
    - dict: Confusion matrix and key performance metrics.
    """
    if not results:
        return {"error": "No results to aggregate"}

    # Count basic classification outcomes
    tp = sum(r["true_positive"] for r in results)
    tn = sum(r["true_negative"] for r in results)
    fp = sum(r["false_positive"] for r in results)
    fn = sum(r["false_negative"] for r in results)

    # Calculate performance metrics
    total_samples = tp + tn + fp + fn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "performance_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    }
