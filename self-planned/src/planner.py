import json
from pydantic_ai import Agent
from models import Plan
from typing import Dict, Any, Optional


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
        "openai:o3-mini",
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
        return json.loads(result.output)
    except (json.JSONDecodeError, ValueError):
        # Fallback to a basic but structured schema
        return {
            "type": "object",
            "properties": {write_key: {"type": "object"} for write_key in writes},
            "required": writes
        }


async def detect_algorithm(task_description: str) -> str:
    """Detect if a specific algorithm is mentioned in the task description."""

    algorithm_detector = Agent(
        "openai:gpt-4o-mini",
        output_type=str,
        system_prompt="""
Extract the main algorithm/method mentioned in the task description.
Return ONLY the algorithm name as it would commonly appear in academic literature.

Rules:
- Look for specific named algorithms, not generic terms
- Return the most standard academic name for the algorithm
- If an algorithm has common abbreviations, include both
- Ignore generic terms like "reasoning", "analysis", "method", "approach"
- If no specific algorithm is mentioned, return "none"

Return just the algorithm name, nothing else.
"""
    )

    result = await algorithm_detector.run(task_description)
    return result.output.strip()


async def retrieve_algorithm_knowledge(algorithm_name: str) -> str:
    """Retrieve canonical algorithmic knowledge for the specified algorithm."""

    knowledge_retriever = Agent(
        "openai:o3-mini",
        output_type=str,
        system_prompt="""
You are an algorithm expert. Provide the canonical mathematical stages/steps for the requested algorithm as they appear in academic literature.

Format your response as:
ALGORITHM: [name]
DEFINITION: [brief mathematical definition]
CANONICAL STAGES:
1. [Stage name]: [mathematical description]
2. [Stage name]: [mathematical description]
...
KEY MATHEMATICAL OBJECTS: [list the main data structures/objects manipulated]

Be precise and focus on the algorithmic structure, not explanations or applications.
"""
    )

    knowledge_prompt = f"Describe the canonical stages of {algorithm_name}"
    result = await knowledge_retriever.run(knowledge_prompt)
    return result.output


def create_generic_planner() -> Agent[None, Plan]:
    """Create a generic planner for non-algorithmic tasks."""

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

    return Agent(
        "openai:o3-mini",
        output_type=Plan,
        system_prompt=system_prompt
    )


def create_algorithm_informed_planner(algorithm_knowledge: str) -> Agent[None, Plan]:
    """Create planner informed with specific algorithm knowledge."""

    system_prompt = f"""
You are a planning model that decomposes algorithmic tasks into stages.
Return ONLY a JSON object that parses into the provided `Plan` type. Do not include prose, comments, or markdown.

ALGORITHM KNOWLEDGE:
{algorithm_knowledge}

CRITICAL: Use this algorithmic knowledge to create stages that implement the canonical mathematical phases of the algorithm. Do NOT create generic reasoning stages like "parse input" or "evaluate hypothesis".

Algorithmic Planning Guidelines:
- Each stage should correspond to a specific algorithmic phase from the knowledge above
- Stages should build mathematically on each other following the algorithm's structure
- Output mathematical objects (graphs, matrices, sets, relations) as JSON structures
- Each stage should implement one algorithmic transformation/computation
- Use the exact terminology and concepts from the algorithm knowledge
- Prefer number of stages matching the algorithm's natural mathematical decomposition

Prompt Template Guidelines:
- Create DETAILED, INFORMATIVE prompt templates that maximize execution quality
- Include specific instructions on HOW to perform the algorithmic step
- Mention potential edge cases, common pitfalls, and validation criteria
- Provide clear guidance on expected input formats and output structures
- Include mathematical definitions and constraints relevant to the step
- Be comprehensive rather than concise - detailed prompts lead to better execution

Contract:
- Each stage defines:
  - reads[]: context keys it expects (subset of keys available so far),
  - writes[]: new/updated context keys it will produce (non-empty),
  - prompt_template: detailed template using only {{placeholders}} that exist in reads[],
  - output_schema: strict JSON Schema describing exactly the keys you write,
- Context model:
  - The initial context contains ONLY the keys named by the caller.
  - After each stage, ONLY keys in writes[] may be added/updated in context.
- Reads/writes discipline:
  - reads[] must be a subset of available keys (initial keys ∪ prior writes).
  - Every writes[] key MUST appear in the stage's output JSON (per output_schema).
- The plan MUST set final_key to the context key that represents the final answer/result.
- All stage outputs must be STRICT JSON (no extra text).
"""

    return Agent(
        "openai:o3-mini",
        output_type=Plan,
        system_prompt=system_prompt
    )


async def create_planner(task_description: Optional[str] = None) -> Agent[None, Plan]:
    """Create an algorithm-aware planner that adapts based on the task description."""

    if task_description is None:
        return create_generic_planner()

    # Step 1: Detect if an algorithm is mentioned
    algorithm_name = await detect_algorithm(task_description)

    if algorithm_name == "none":
        print("🔍 No specific algorithm detected - using generic planner")
        return create_generic_planner()

    # Step 2: Retrieve algorithm knowledge
    print(f"🔍 Detected algorithm: {algorithm_name}")
    print("📚 Retrieving algorithm knowledge...")

    try:
        algorithm_knowledge = await retrieve_algorithm_knowledge(algorithm_name)
        print(f"✅ Algorithm knowledge retrieved:")
        print("-" * 60)
        print(algorithm_knowledge)
        print("-" * 60)

        # Step 3: Create algorithm-informed planner
        return create_algorithm_informed_planner(algorithm_knowledge)

    except Exception as e:
        print(f"⚠️ Failed to retrieve algorithm knowledge: {e}")
        print("🔄 Falling back to generic planner")
        return create_generic_planner()
