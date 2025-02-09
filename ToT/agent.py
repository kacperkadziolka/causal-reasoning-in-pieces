from typing import Optional

from ToT.implementation import search


class CausalSkeletonToT:
    def __init__(self, max_steps: int, threshold: int) -> None:
        self.max_steps = max_steps + 1
        self.threshold = threshold

    def run(self, initial_prompt: str) -> Optional[str]:
        """
        Runs the ToT search starting from the initial prompt (the system instructions combined with few-shot examples).
        Returns the final output (complete chain-of-thought).
        """
        final_state = search(initial_state=initial_prompt,
                             current_step=1,
                             max_steps=self.max_steps,
                             threshold=self.threshold
                             )
        return final_state
