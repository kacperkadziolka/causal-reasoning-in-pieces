import ast
import json
from typing import Dict, Any
from pydantic_ai import Agent
from models import Stage, Plan


def create_executor() -> Agent[None, str]:
    """Create the executor agent that runs individual stages."""

    system_prompt = """
You execute a specific stage of a decomposed reasoning workflow.
You will receive a rendered prompt template with context data.
Return ONLY valid JSON that matches the given output_schema exactly.
Focus only on the specific task described in the prompt.
Do not include explanations, markdown, or additional text - only the raw JSON.
"""

    executor = Agent(
        "openai:gpt-4o-mini",
        output_type=str,
        system_prompt=system_prompt
    )

    return executor


async def run_stage(stage: Stage, context: Dict[str, Any]) -> Dict[str, Any]:
    # Create executor
    executor = create_executor()

    # Get the data this stage needs to read
    read_data = {key: context.get(key) for key in stage.reads}

    # Render the prompt template with the read data
    try:
        rendered_prompt = stage.prompt_template.format(**read_data)
    except KeyError as e:
        raise ValueError(f"Stage '{stage.id}' tried to read key {e} that doesn't exist in context")

    # Add schema information to guide the output
    prompt_with_schema = f"{rendered_prompt}\n\nOutput JSON Schema:\n{stage.output_schema}"

    print(f"  Running stage: {stage.id}")
    print(f"  Prompt: {rendered_prompt}")

    # Execute the stage
    result = await executor.run(prompt_with_schema)
    json_output = result.output

    # Parse the JSON response
    try:
        stage_output = json.loads(json_output)
    except json.JSONDecodeError:
        try:
            stage_output = ast.literal_eval(json_output)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Stage '{stage.id}' returned invalid JSON/dict: {e}\nOutput: {json_output}")

    # Validate that all required keys are present
    for key in stage.writes:
        if key not in stage_output:
            raise ValueError(f"Stage '{stage.id}' did not produce required output key '{key}'")

    return stage_output


async def run_plan(plan: Plan, initial_context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a complete plan by running all stages sequentially."""

    print(f"\n=== EXECUTING PLAN ({len(plan.stages)} stages) ===")

    # Start with the initial context
    context = dict(initial_context)

    # Run each stage sequentially
    for i, stage in enumerate(plan.stages, 1):
        print(f"\n--- Stage {i}/{len(plan.stages)}: {stage.id} ---")
        print(f"Reads: {stage.reads}")
        print(f"Writes: {stage.writes}")

        try:
            # Execute the stage
            stage_output = await run_stage(stage, context)

            # Update context with stage outputs
            for key in stage.writes:
                context[key] = stage_output[key]
                print(f"  ✓ Wrote '{key}': {str(stage_output[key])[:100]}...")

        except Exception as e:
            print(f"  ✗ Error in stage '{stage.id}': {e}")
            raise

    print("\n=== PLAN EXECUTION COMPLETE ===")
    return context
