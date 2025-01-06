import json
import os

import pytest

from CoT.answer_extractor import extract_edges_incident_format


def load_test_cases(json_file_path):
    """
    Loads test cases from a JSON file. Each test case is a dictionary
    containing:
      - name
      - answer
      - expected_edges or expected_error_substring
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


@pytest.mark.parametrize("test_case", load_test_cases(os.path.join(os.path.dirname(__file__), "test_cases.json")))
def test_extract_edges_incident_format(test_case):
    answer = test_case["answer"]

    # If the JSON has 'expected_error_substring', we test that an error is raised
    if "expected_error_substring" in test_case:
        expected_error_substring = test_case["expected_error_substring"]
        with pytest.raises(RuntimeError) as exc_info:
            _ = extract_edges_incident_format(answer)
        assert expected_error_substring in str(exc_info.value), (
            f"Error message did not contain '{expected_error_substring}'. "
            f"Actual message: {exc_info.value}"
        )

    # Otherwise, we assume we have an 'expected_edges' list of pairs
    else:
        expected_edges_list = test_case["expected_edges"]
        expected_edges = {tuple(sorted(pair)) for pair in expected_edges_list}

        edges = extract_edges_incident_format(answer)
        assert edges == expected_edges, f"For test '{test_case.get('name')}', expected {expected_edges}, got {edges}"
