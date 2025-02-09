from ToT.agent import CausalSkeletonToT
from ToT.utils import prompts


def main() -> None:
    initial_prompt: str = prompts["initial_prompt"]

    agent: CausalSkeletonToT = CausalSkeletonToT(max_steps=4, threshold=5)
    final_output: str = agent.run(initial_prompt)
    if final_output:
        print("Final causal undirected skeleton:")
        print(final_output)
    else:
        print("Failed to generate a valid solution.")


if __name__ == "__main__":
    main()
