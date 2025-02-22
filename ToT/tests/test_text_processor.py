import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest

from ToT.text_processor import extract_edges


def load_test_cases():
    test_data_path = os.path.join(os.path.dirname(__file__), "data/test_text_processor.json")
    with open(test_data_path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            # Single test case
            test_data = [data]
        else:
            # Multiple test cases
            test_data = data
    return test_data


@pytest.mark.parametrize("test_case", load_test_cases())
def test_extract_edges(test_case):
    input_text = test_case["input"]
    expected_edges_list = test_case["expected_edges"]
    expected_edges = {tuple(sorted(edge)) for edge in expected_edges_list}

    result_edges = extract_edges(input_text)
    assert result_edges == expected_edges
