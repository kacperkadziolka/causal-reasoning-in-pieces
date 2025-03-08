import json
import pytest
from CoT_reasoning_models.answer_extractor import extract_edges_json

# Load test cases from JSON file
with open('test_cases.json', 'r') as file:
    test_cases = json.load(file)


@pytest.mark.parametrize("test_case", test_cases, ids=[case["description"] for case in test_cases])
def test_extract_edges_json(test_case):
    answer = test_case["answer"]
    expected_edges = set(tuple(sorted(edge)) for edge in test_case["expected_edges"])
    assert extract_edges_json(answer) == expected_edges


@pytest.mark.parametrize("test_case", test_cases, ids=[case["description"] for case in test_cases])
def test_extract_edges_json_no_exceptions(test_case):
    """Test that extract_edges_json doesn't throw any exceptions."""
    answer = test_case["answer"]
    try:
        extract_edges_json(answer)
    except Exception as e:
        pytest.fail(f"extract_edges_json raised {type(e).__name__} unexpectedly: {e}")
