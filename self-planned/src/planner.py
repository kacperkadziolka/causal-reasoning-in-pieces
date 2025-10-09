from pydantic_ai import Agent
from .models import Plan
from typing import Dict, Any


def is_schema_too_generic(schema: Dict[str, Any]) -> bool:
    """Check if schema is too vague or generic"""
    if not isinstance(schema, dict):
        return True

    for key, value in schema.items():
        # Check for completely empty schemas
        if value == {} or value == "object" or value == "string" or value == "array":
            return True
        # Check for schemas without proper structure
        if isinstance(value, dict) and not value:
            return True

    return False


async def refine_schema(stage_id: str, writes: list[str], prompt_template: str) -> Dict[str, Any]:
    """Refine a generic schema to be more specific"""

    schema_refiner = Agent(
        "openai:gpt-4o-mini",
        output_type=str,
        system_prompt="""
You are a JSON Schema expert. Given a stage description, create a SPECIFIC JSON Schema.

CRITICAL: The output must have top-level properties matching the "writes" keys exactly.

Rules:
- The schema MUST have top-level properties for each key in "writes"
- NO generic types like "object", "string", "array" without structure
- Define exact property names and types for the content inside each write key
- Include "required" fields listing all write keys
- For arrays, specify "items" structure
- For objects, specify "properties"

Example: If writes = ["graph_data"], the schema should be:
{
  "type": "object",
  "properties": {
    "graph_data": {
      "type": "object",
      "properties": { ... detailed structure ... }
    }
  },
  "required": ["graph_data"]
}

Return ONLY valid JSON Schema as a JSON object, no markdown or explanations.
"""
    )

    refinement_prompt = f"""
Stage: {stage_id}
Writes: {writes}
Purpose: {prompt_template}

Create a specific JSON Schema where the top-level properties are exactly: {writes}
Each property should have a detailed structure appropriate for the stage purpose.
The schema must match what the executor expects based on the "writes" keys.
"""

    result = await schema_refiner.run(refinement_prompt)

    try:
        import json
        return json.loads(result.output)
    except:
        # Fallback to a basic but structured schema
        return {
            "type": "object",
            "properties": {write_key: {"type": "object"} for write_key in writes},
            "required": writes
        }


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
