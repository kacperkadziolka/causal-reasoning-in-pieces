import json

import pytest

from CoT_reasoning_models.meek_rules import apply_meek_rules

with open('test_meek_rules.json', 'r') as file:
    test_cases = json.load(file)

@pytest.mark.parametrize("test_case", test_cases, ids=[case["name"] for case in test_cases])
def test_apply_meek_rules(test_case):
    """Test the application of Meek rules with various test cases"""
    # Convert input to proper format
    skeleton = [tuple(edge) for edge in test_case["skeleton"]]
    v_structures = [tuple(v) for v in test_case["v_structures"]]
    expected = {tuple(edge) for edge in test_case["expected"]}

    # Apply Meek rules
    result = set(apply_meek_rules(skeleton, v_structures))

    # Check results
    assert result == expected, f"Failed {test_case['name']}: Expected {sorted(expected)}, got {sorted(result)}"

    # Additional descriptive output for debugging
    print(f"✓ PASSED: {test_case['name']} - {test_case['description']}")
