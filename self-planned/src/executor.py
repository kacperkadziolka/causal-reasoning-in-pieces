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
        "openai:o3-mini",
        output_type=str,
        system_prompt=system_prompt
    )

    return executor


async def run_stage(stage: Stage, context: Dict[str, Any]) -> Dict[str, Any]:
    import time

    executor = create_executor()

    # Get the data this stage needs to read
    read_data = {key: context.get(key) for key in stage.reads}

    print(f"\n🔄 {stage.id}: {stage.reads} → {stage.writes}", end=" ")

    start_time = time.time()

    # Render the prompt template with the read data
    try:
        # First, escape any braces that aren't actual placeholders
        escaped_template = stage.prompt_template

        # Find all placeholders that should be replaced (those that match keys in read_data)
        import re
        actual_placeholders = set(read_data.keys())

        # Replace all {text} that are NOT actual placeholders with {{text}}
        def escape_non_placeholders(match):
            placeholder = match.group(1)
            if placeholder in actual_placeholders:
                return match.group(0)  # Keep as {placeholder}
            else:
                return '{{' + placeholder + '}}'  # Escape as {{placeholder}}

        escaped_template = re.sub(r'\{([^}]+)\}', escape_non_placeholders, escaped_template)

        rendered_prompt = escaped_template.format(**read_data)
    except KeyError as e:
        raise ValueError(f"Stage '{stage.id}' tried to read key {e} that doesn't exist in context")

    # Add schema information to guide the output
    prompt_with_schema = f"{rendered_prompt}\n\nOutput JSON Schema:\n{json.dumps(stage.output_schema, indent=2)}"

    # Execute the stage
    result = await executor.run(prompt_with_schema)
    json_output = result.output

    # Parse the JSON response
    try:
        stage_output = json.loads(json_output)
        print("✅", end="")
    except json.JSONDecodeError:
        try:
            stage_output = ast.literal_eval(json_output)
            print("✅", end="")
        except (ValueError, SyntaxError) as e:
            print("❌")
            raise ValueError(f"Stage '{stage.id}' returned invalid JSON/dict: {e}")

    # Validate that all required keys are present
    missing_keys = [key for key in stage.writes if key not in stage_output]
    if missing_keys:
        print(f"❌ Missing: {missing_keys}")
        raise ValueError(f"Stage '{stage.id}' did not produce required output keys: {missing_keys}")

    # Calculate execution time and output metrics
    execution_time = time.time() - start_time

    # Get size info and preview for outputs
    output_info = []
    output_preview = []

    for key in stage.writes:
        value = stage_output[key]
        if isinstance(value, dict):
            output_info.append(f"{key}={len(value)} keys")
            # Show some key names for dicts
            if value:
                preview_keys = list(value.keys())[:2]
                if len(value) > 2:
                    preview_keys.append("...")
                output_preview.append(f"{key}:[{', '.join(str(k) for k in preview_keys)}]")
        elif isinstance(value, list):
            output_info.append(f"{key}={len(value)} items")
            # Show first few items for lists
            if value:
                preview_items = value[:2] if len(value) >= 2 else value
                preview_str = str(preview_items)[1:-1]  # Remove brackets
                if len(value) > 2:
                    preview_str += ", ..."
                output_preview.append(f"{key}:[{preview_str}]")
        elif isinstance(value, str):
            output_info.append(f"{key}={len(value)} chars")
            # Show truncated string
            preview = value[:30] + "..." if len(value) > 30 else value
            output_preview.append(f"{key}:'{preview}'")
        elif isinstance(value, bool):
            output_info.append(f"{key}=bool")
            output_preview.append(f"{key}:{value}")
        else:
            output_info.append(f"{key}={type(value).__name__}")
            output_preview.append(f"{key}:{str(value)[:20]}")

    print(f" ({execution_time:.1f}s, {', '.join(output_info)})")

    # Show output preview on next line with indentation
    if output_preview:
        print(f"     → {', '.join(output_preview)}")

    # Show full content of outputs
    print(f"     📄 Full Output:")
    for key in stage.writes:
        value = stage_output[key]
        print(f"       {key}: {value}")

    return stage_output


async def run_plan(plan: Plan, initial_context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a complete plan by running all stages sequentially."""
    import time

    plan_start_time = time.time()
    print(f"\n🎯 Executing {len(plan.stages)} stages:")
    context = dict(initial_context)

    # Run each stage sequentially
    for i, stage in enumerate(plan.stages, 1):
        try:
            # Execute the stage
            stage_output = await run_stage(stage, context)

            # Update context with stage outputs
            for key in stage.writes:
                context[key] = stage_output[key]

        except Exception as e:
            print(f"❌ ERROR in {stage.id}: {e}")
            raise

    total_time = time.time() - plan_start_time
    final_result = context.get(plan.final_key, '<<NOT_FOUND>>')
    print(f"\n🎯 Completed in {total_time:.1f}s → {final_result}")

    return context
