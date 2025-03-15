import json
import pytest
from CoT_reasoning_models.answer_extractor import extract_edges_json, extract_separation_sets, find_v_structures

with open('test_cases.json', 'r') as file:
    test_cases = json.load(file)

with open('test_separation_sets.json', 'r') as file:
    separation_test_cases = json.load(file)

with open('test_v_structures.json', 'r') as file:
    v_structure_test_cases = json.load(file)


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


@pytest.mark.parametrize("test_case", separation_test_cases,
                         ids=[case["description"] for case in separation_test_cases])
def test_extract_separation_sets(test_case):
    premise = test_case["premise"]
    # Convert string representations back to proper data structures
    expected_sep_sets = {}
    for pair_str, sep_set_list in test_case["expected_separation_sets"].items():
        pair = frozenset(pair_str.split(','))
        expected_sep_sets[pair] = set(sep_set_list)

    actual_sep_sets = extract_separation_sets(premise)

    # Compare the two dictionaries
    assert len(actual_sep_sets) == len(expected_sep_sets), "Number of separation sets doesn't match"

    for pair, sep_set in actual_sep_sets.items():
        assert pair in expected_sep_sets, f"Pair {pair} not found in expected separation sets"
        assert sep_set == expected_sep_sets[
            pair], f"Separation set for {pair} doesn't match: expected {expected_sep_sets[pair]}, got {sep_set}"


@pytest.mark.parametrize("test_case", v_structure_test_cases,
                         ids=[case["description"] for case in v_structure_test_cases])
def test_find_v_structures(test_case):
    # Parse the edges from string representation to tuples
    skeleton_edges = [tuple(edge) for edge in test_case["skeleton_edges"]]

    # Convert separation sets from string representation to proper data structures
    separation_sets = {}
    for pair_str, sep_set_list in test_case["separation_sets"].items():
        pair = frozenset(pair_str.split(','))
        separation_sets[pair] = set(sep_set_list)

    # Parse expected v-structures as tuples
    expected_v_structures = [tuple(v_struct) for v_struct in test_case["expected_v_structures"]]

    # Run the function
    result = find_v_structures(skeleton_edges, separation_sets)

    # Normalize v-structures by ensuring the middle node is fixed and the endpoints are sorted
    def normalize_v_structure(v_struct):
        if len(v_struct) != 3:
            return v_struct
        # Keep the middle node (collider) fixed but sort the endpoints
        endpoints = sorted([v_struct[0], v_struct[2]])
        return (endpoints[0], v_struct[1], endpoints[1])

    normalized_result = [normalize_v_structure(v) for v in result]
    normalized_expected = [normalize_v_structure(v) for v in expected_v_structures]

    # Sort the normalized results for consistent comparison
    sorted_result = sorted(normalized_result)
    sorted_expected = sorted(normalized_expected)

    assert sorted_result == sorted_expected, f"V-structure mismatch: expected {sorted_expected}, got {sorted_result}"
