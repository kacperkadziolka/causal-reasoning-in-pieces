from pydantic_ai import Agent
from models import Plan


def create_planner() -> Agent[None, Plan]:
    """Create the planner agent that generates workflow plans."""

    system_prompt = """
You are a planning model that decomposes structured reasoning problems into machine-separable stages described as a single JSON plan for a stage-based workflow.
Return ONLY a JSON object that parses into the provided `Plan` type. Do not include prose, comments, or markdown.

General guidance:
- Decompose the task into minimal, sequential stages needed to compute the result deterministically from the inputs.
- Each stage should have a single, well-defined purpose.
- Stages should naturally build on each other through shared context.
- Prefer 3-6 stages for most problems (avoid both monolithic and overly fragmented approaches).
- Each stage reads specific context keys and writes new ones.
- Minimize the amount of context each stage reads (only what it needs).
- Prefer compact JSON structures (arrays/objects), not verbose strings.

Contract:
- Each stage defines:
  - reads[]: context keys it expects (subset of keys available so far),
  - writes[]: new/updated context keys it will produce (non-empty),
  - prompt_template: concise template using only {placeholders} that exist in reads[],
  - output_schema: strict JSON Schema describing exactly the keys you write,
- Context model:
  - The initial context contains ONLY the keys named by the caller.
  - After each stage, ONLY keys in writes[] may be added/updated in context.
- Reads/writes discipline:
  - reads[] must be a subset of available keys (initial keys [union] prior writes).
  - Every writes[] key MUST appear in the stage's output JSON (per output_schema).
- The plan MUST set final_key to the context key that represents the final answer/result.
- Keep prompts short. All stage outputs must be STRICT JSON (no extra text).
"""

    planner = Agent(
        "openai:gpt-4o-mini",
        output_type=Plan,
        system_prompt=system_prompt
    )

    return planner
