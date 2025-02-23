import logging
import os
from typing import Optional

from ToT.llm.llm_factory import get_llm_model
from ToT.utils import prompts, log_directory, evaluate_prompts, config

run_id: str = config.get("run_id", "default_run")
logging.basicConfig(
    filename=os.path.join(log_directory, f"ToT-{run_id}.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def generate_thoughts(state: str, next_step_instruction: str, next_step_example: str, k: int) -> list[str]:
    """
    Given the current state (chain-of-thought) and instruction for the next step,
    generate k candidate next thoughts.
    """
    generate_prompt_template: str = prompts["generate_prompt_template"]
    generate_prompt: str = generate_prompt_template.format(
        current_state=state,
        next_step_instruction=next_step_instruction,
        next_step_example=next_step_example,
    )
    system_prompt: str = prompts["system_prompt"]

    logging.info("generate_thoughts - Sending prompt to model:\n%s", generate_prompt)

    llm_model = get_llm_model()
    responses =  llm_model.generate(
        user_prompt=generate_prompt,
        system_prompt=system_prompt,
        num_samples=k
    )

    for i, response in enumerate(responses, start=1):
        logging.info("generate_thoughts - Candidate %d:\n%s", i, response)

    return responses


def evaluate_state(previous_state: str, recent_step: str, instructions_constraints: str, step_number: int) -> int:
    """
    Evaluates the given state by asking the model to score it on how well it satisfies the constraints.
    Returns a numeric score.
    """
    if step_number == 3:
        evaluate_prompt_template: str = evaluate_prompts["step_3_template"]
    else:
        evaluate_prompt_template: str = prompts["evaluate_prompt_template"]

    evaluate_prompt: str = evaluate_prompt_template.format(
        previous_state=previous_state,
        recent_step=recent_step,
        instruction_constraints=instructions_constraints
    )

    logging.info("evaluate_state - Sending prompt to model:\n%s", evaluate_prompt)

    llm_model = get_llm_model()
    response = llm_model.generate(
        user_prompt=evaluate_prompt,
        num_samples=1
    )[0]

    logging.info("evaluate_state - Model response:\n%s", response)

    try:
        score = int(response)
    except ValueError:
        score = 0

    return score


def is_terminal(max_steps: int, current_step: int) -> bool:
    """
    Determines whether a state is terminal (i.e., a complete solution) based on the current step.
    """
    return current_step >= max_steps


def search(initial_state: str, current_step: int, max_steps: int, threshold: int) -> Optional[str]:
    """
    Recursively search for a promising final state using DFS.
    """
    if is_terminal(max_steps, current_step):
        return initial_state

    # Get the next step instruction
    steps_instructions: list[str] = prompts["steps_instructions"]
    next_step_instruction: str = steps_instructions[current_step]

    # Get the next step example
    steps_examples: list[str] = prompts["steps_examples"]
    next_step_example: str = steps_examples[current_step]

    # Get the next step header
    steps_headers: list[str] = prompts["steps_headers"]
    next_step_header: str = steps_headers[current_step]

    # Generate candidate next thoughts
    candidates: list[str] = generate_thoughts(
        state=initial_state,
        next_step_instruction=next_step_instruction,
        next_step_example=next_step_example,
        k=3
    )

    # Evaluate each candidate state
    evaluated_candidates: list[tuple[str, int]] = []
    for candidate in candidates:
        candidate_step_text = prompts["step_template"].format(
            step_number=next_step_header,
            candidate_step=candidate
        )
        new_state: str = f"{initial_state}\n\n{candidate_step_text}"

        score: int = evaluate_state(
            previous_state=initial_state,
            recent_step=candidate,
            instructions_constraints=next_step_instruction,
            step_number=current_step
        )
        evaluated_candidates.append((new_state, score))

    # Filter out candidates that do not meet the threshold
    evaluated_candidates = [candidate for candidate in evaluated_candidates if candidate[1] >= threshold]
    if not evaluated_candidates:
        logging.info("No candidate at step %d meets the threshold. Backtracking...", current_step)
        return None

    # Sort candidates by score (highest first)
    evaluated_candidates.sort(key=lambda x: x[1], reverse=True)

    # Try each candidate in turn
    for new_state, score in evaluated_candidates:
        logging.info("Trying candidate at step %d with score %d", current_step, score)
        result = search(
            initial_state=new_state,
            current_step=current_step + 1,
            max_steps=max_steps,
            threshold=threshold
        )
        if result is not None:
            return result

    # If none of the candidates at the current level lead to a final solution
    return None
