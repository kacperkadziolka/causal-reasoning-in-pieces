import logging
from typing import Optional

from ToT.utils import prompts, call_llm, get_openai_client


logging.basicConfig(
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

    responses =  call_llm(
        client=get_openai_client(),
        user_prompt=generate_prompt,
        system_prompt=system_prompt,
        num_samples=k
    )

    for i, response in enumerate(responses, start=1):
        logging.info("generate_thoughts - Candidate %d:\n%s", i, response)

    return responses


def evaluate_state(previous_state: str, recent_step: str, instructions_constraints: str) -> int:
    """
    Evaluates the given state by asking the model to score it on how well it satisfies the constraints.
    Returns a numeric score.
    """
    evaluate_prompt_template: str = prompts["evaluate_prompt_template"]
    evaluate_prompt: str = evaluate_prompt_template.format(
        previous_state=previous_state,
        recent_step=recent_step,
        instruction_constraints=instructions_constraints
    )

    logging.info("evaluate_state - Sending prompt to model:\n%s", evaluate_prompt)

    response = call_llm(client=get_openai_client(), user_prompt=evaluate_prompt, num_samples=1)[0]

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

    # Generate candidate next thoughts
    candidates: list[str] = generate_thoughts(state=initial_state,
                                              next_step_instruction=next_step_instruction,
                                              next_step_example=next_step_example,
                                              k=3)

    # Evaluate each candidate state
    evaluated_candidates: list[tuple[str, int]] = []
    for candidate in candidates:
        # Append candidate thought to state
        # TODO: Make this more robust
        new_state: str = f"{initial_state} \n\n {candidate}"

        score: int = evaluate_state(
            previous_state=initial_state,
            recent_step=candidate,
            instructions_constraints=next_step_instruction
        )
        evaluated_candidates.append((new_state, score))

    # Evaluate candidates above threshold
    evaluated_candidates = [candidate for candidate in evaluated_candidates if candidate[1] >= threshold]
    if not evaluated_candidates:
        logging.info("No candidate state meets the threshold. Backtracking...")
        return None

    # Sort candidates by score and explore the best one
    best_candidate, best_score = max(evaluated_candidates, key=lambda x: x[1])
    logging.info("Best candidate at step %d with score %d", current_step, best_score)

    # Recursive call for the next step
    return search(initial_state=best_candidate, current_step=current_step + 1, max_steps=max_steps, threshold=threshold)
