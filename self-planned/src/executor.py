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
    executor = create_executor()

    # Get the data this stage needs to read
    read_data = {key: context.get(key) for key in stage.reads}

    print(f"\n🔄 EXECUTING STAGE: {stage.id}")
    print(f"📖 Context keys available: {list(context.keys())}")
    print(f"📥 Reading from context: {stage.reads}")
    print(f"📤 Will write to context: {stage.writes}")

    # Show what data is being read
    if stage.reads:
        print("📊 Read data:")
        for key in stage.reads:
            value = read_data[key]
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}... (truncated)")
            else:
                print(f"  {key}: {value}")

    # Render the prompt template with the read data
    try:
        rendered_prompt = stage.prompt_template.format(**read_data)
    except KeyError as e:
        raise ValueError(f"Stage '{stage.id}' tried to read key {e} that doesn't exist in context")

    # Add schema information to guide the output
    prompt_with_schema = f"{rendered_prompt}\n\nOutput JSON Schema:\n{json.dumps(stage.output_schema, indent=2)}"

    print("\n📝 FULL PROMPT SENT TO LLM:")
    print("-" * 80)
    print(prompt_with_schema)
    print("-" * 80)

    # Execute the stage
    print(f"\n⏳ Calling LLM for stage '{stage.id}'...")
    result = await executor.run(prompt_with_schema)
    json_output = result.output

    print("\n📤 RAW LLM RESPONSE:")
    print("-" * 80)
    print(json_output)
    print("-" * 80)

    # Parse the JSON response
    try:
        stage_output = json.loads(json_output)
        print("✅ Successfully parsed JSON response")
    except json.JSONDecodeError:
        try:
            stage_output = ast.literal_eval(json_output)
            print("✅ Successfully parsed response using ast.literal_eval")
        except (ValueError, SyntaxError) as e:
            print("❌ Failed to parse response as JSON or dict")
            raise ValueError(f"Stage '{stage.id}' returned invalid JSON/dict: {e}\nOutput: {json_output}")

    # Show parsed output
    print("\n📋 PARSED STAGE OUTPUT:")
    for key, value in stage_output.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}... (truncated)")
        else:
            print(f"  {key}: {value}")

    # Validate that all required keys are present
    for key in stage.writes:
        if key not in stage_output:
            print(f"❌ Missing required output key: '{key}'")
            raise ValueError(f"Stage '{stage.id}' did not produce required output key '{key}'")
        print(f"✅ Found required output key: '{key}'")

    return stage_output


async def run_plan(plan: Plan, initial_context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a complete plan by running all stages sequentially."""

    print(f"\n🎯 === EXECUTING PLAN ({len(plan.stages)} stages) ===")
    print(f"🔑 Final result will be read from key: '{plan.final_key}'")

    # Start with the initial context
    context = dict(initial_context)

    print("\n🏁 INITIAL CONTEXT:")
    print(f"📋 Keys: {list(context.keys())}")
    for key, value in context.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}... (truncated)")
        else:
            print(f"  {key}: {value}")

    # Run each stage sequentially
    for i, stage in enumerate(plan.stages, 1):
        print(f"\n🔢 === STAGE {i}/{len(plan.stages)}: {stage.id} ===")

        print(f"\n📊 CONTEXT BEFORE STAGE {i}:")
        print(f"📋 Available keys: {list(context.keys())}")

        try:
            # Execute the stage
            stage_output = await run_stage(stage, context)

            # Update context with stage outputs
            print("\n🔄 UPDATING CONTEXT WITH STAGE OUTPUT:")
            for key in stage.writes:
                old_value = context.get(key, "<<NOT_PRESENT>>")
                new_value = stage_output[key]
                context[key] = new_value

                print(f"  🔄 Key '{key}':")
                print(f"    Before: {old_value}")
                print(f"    After:  {new_value}")

            print(f"\n📊 CONTEXT AFTER STAGE {i}:")
            print(f"📋 Available keys: {list(context.keys())}")

            # Show summary of context evolution
            print("\n📈 CONTEXT EVOLUTION SUMMARY:")
            for key, value in context.items():
                if isinstance(value, dict):
                    print(f"  {key}: {{dict with {len(value)} keys}}")
                elif isinstance(value, list):
                    print(f"  {key}: [list with {len(value)} items]")
                elif isinstance(value, str):
                    if len(value) > 100:
                        print(f"  {key}: \"{value[:100]}...\" (truncated)")
                    else:
                        print(f"  {key}: \"{value}\"")
                else:
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"❌ ERROR in stage '{stage.id}': {e}")
            print(f"📊 Context at time of error: {list(context.keys())}")
            raise

    print("\n🎉 === PLAN EXECUTION COMPLETE ===")
    print(f"🔑 Final context keys: {list(context.keys())}")
    print(f"🎯 Final answer key '{plan.final_key}': {context.get(plan.final_key, '<<NOT_FOUND>>')}")

    return context
