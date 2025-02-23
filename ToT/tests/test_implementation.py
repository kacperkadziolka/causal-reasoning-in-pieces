import os

import pytest

from ToT.tests.conftest import get_test_model

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import yaml

from ToT.llm.llm_factory import get_llm_model


@pytest.mark.skip
def test_evaluate_state_human_evaluation():
    test_data_path = os.path.join(os.path.dirname(__file__), "data/test_evaluate_state.yaml")
    with open(test_data_path, "r") as f:
        test_cases = yaml.safe_load(f)

    # Get the LLM model instance.
    llm_model = get_llm_model()

    for idx, test_case in enumerate(test_cases, start=1):
        description = test_case.get("description", f"Test case {idx}")
        user_prompt = test_case["user_prompt"]

        # Call the generate function with the provided user prompt.
        responses = llm_model.generate(user_prompt=user_prompt, num_samples=1)
        response_text = responses[0] if responses else "<No response>"

        print("============================================")
        print(f"Test Case {idx}: {description}")
        print(f"User Prompt: {user_prompt}")
        print(f"LLM Response: {response_text}")
        print("============================================\n")


@pytest.mark.skip
def test_generate_state_human_evaluation():
    test_data_path = os.path.join(os.path.dirname(__file__), "data/test_generate_state.yaml")
    with open(test_data_path, "r") as f:
        test_cases = yaml.safe_load(f)

    # Get the LLM model instance.
    llm_model = get_test_model()

    for idx, test_case in enumerate(test_cases, start=1):
        description = test_case.get("description", f"Test case {idx}")
        user_prompt = test_case["user_prompt"]

        # Call the generate function with the provided user prompt.
        responses = llm_model.generate(user_prompt=user_prompt, num_samples=1)
        response_text = responses[0] if responses else "<No response>"

        print("============================================")
        print(f"Test Case {idx}: {description}")
        print(f"User Prompt: {user_prompt}")
        print(f"LLM Response: {response_text}")
        print("============================================\n")


@pytest.mark.skip
def test_single_prompt():
    test_data_path = os.path.join(os.path.dirname(__file__), "data/single_prompt.yaml")
    with open(test_data_path, "r") as f:
        test_cases = yaml.safe_load(f)

    # Get the LLM model instance.
    llm_model = get_llm_model()

    for idx, test_case in enumerate(test_cases, start=1):
        description = test_case.get("description", f"Test case {idx}")
        user_prompt = test_case["user_prompt"]

        # Call the generate function with the provided user prompt.
        responses = llm_model.generate(user_prompt=user_prompt, num_samples=1)
        response_text = responses[0] if responses else "<No response>"

        print("============================================")
        print(f"Test Case {idx}: {description}")
        print(f"User Prompt: {user_prompt}")
        print(f"LLM Response: {response_text}")
        print("============================================\n")
