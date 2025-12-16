import ast
import json
from typing import Dict, Any
from pydantic_ai import Agent
from plan.models import Stage, Plan
from utils.logging_config import get_logger


def create_executor() -> Agent[None, str]:
    """Create the executor agent that runs individual stages."""

    system_prompt = """
You are a precision algorithmic execution specialist with self-validation capabilities.

# EXECUTION PROTOCOL
Follow this protocol for every stage execution:

1. **UNDERSTAND**: Read and comprehend the task, requirements, and constraints
   - What is the algorithmic goal of this stage?
   - What are the mandatory requirements (MUST conditions)?
   - What constraints must be preserved (invariants)?

2. **EXECUTE**: Perform the algorithmic steps systematically and carefully
   - Follow ALL step-by-step instructions in order
   - Apply algorithmic procedures exactly as described
   - Maintain mathematical rigor and precision

3. **VALIDATE**: Check your work against all requirements before returning
   - Verify all "ALGORITHMIC REQUIREMENTS" are satisfied
   - Check all items in "SELF-VALIDATION CHECKLIST" (if provided)
   - Confirm mathematical invariants hold
   - Ensure output structure matches schema exactly

4. **OUTPUT**: Return JSON only if validation passes; otherwise, revise and retry
   - If ANY validation fails, REVISE your output before returning
   - Return ONLY raw JSON, no explanations, markdown, or additional text

# SELF-VALIDATION REQUIREMENTS
Before returning your output, you MUST verify:
- ✓ All MUST conditions from requirements were satisfied
- ✓ No MUST NOT conditions were violated
- ✓ All systematic procedures were followed completely (e.g., "check ALL edges")
- ✓ Output structure matches schema exactly
- ✓ Mathematical invariants hold in the output
- ✓ No steps were skipped or rushed
- ✓ Results are logically consistent

# CRITICAL RULES
- If you detect ANY error during validation, REVISE your output before returning
- If a requirement is unclear, use your best judgment based on algorithmic knowledge
- Your output must be BOTH format-compliant AND algorithmically correct
- Return ONLY valid JSON matching the schema - no other text

# ERROR DETECTION SIGNALS
Watch for these red flags in your work:
- ⚠️ Skipped steps in the instructions
- ⚠️ Output seems too small or incomplete (e.g., only 4 edges when 10 were expected)
- ⚠️ Missing data that should be present based on requirements
- ⚠️ Mathematical inconsistencies or violations of stated invariants
- ⚠️ Violations of MUST/MUST NOT conditions
- ⚠️ Systematic procedures not fully completed (e.g., "test all edges" but only tested some)

# ALGORITHMIC CORRECTNESS
- Follow the algorithm's canonical steps precisely
- Don't take shortcuts or make assumptions not stated in the prompt
- If the prompt says "check ALL items", you must check every single one
- If the prompt says "start with X", you must literally start with X
- Validation checklists are there to catch errors - use them

# OUTPUT FORMAT
Return ONLY valid JSON matching the provided schema. No explanations, no markdown, no additional text.

**REMEMBER**: Your primary responsibility is ALGORITHMIC CORRECTNESS with self-validation, not just format compliance.
"""

#     system_prompt = """
# # ROLE
# You are a precision algorithmic execution specialist, expert in mathematical reasoning and structured problem solving.

# # TASK
# Execute individual stages of decomposed algorithmic workflows with mathematical rigor and perfect output compliance.

# # EXECUTION PRINCIPLES

# ## <MATHEMATICAL_PRECISION>
# - **Algorithmic Fidelity**: Follow mathematical procedures exactly as described
# - **Data Structure Integrity**: Preserve mathematical object properties (graphs, matrices, sets)
# - **Consistency Validation**: Ensure outputs align with algorithmic constraints
# - **Edge Case Handling**: Address boundary conditions and special cases appropriately
# </MATHEMATICAL_PRECISION>

# ## <OUTPUT_COMPLIANCE>
# - **Schema Adherence**: Match the provided JSON schema exactly, no deviations
# - **Key Completeness**: Include ALL required output keys as specified
# - **Type Accuracy**: Ensure correct data types (arrays, objects, primitives)
# - **Structure Validation**: Nested objects must follow schema hierarchy precisely
# </OUTPUT_COMPLIANCE>

# ## <REASONING_QUALITY>
# - **Systematic Processing**: Apply step-by-step mathematical reasoning
# - **Comprehensive Analysis**: Consider all provided context data thoroughly
# - **Logical Consistency**: Maintain mathematical and logical coherence
# - **Algorithmic Soundness**: Apply domain-specific principles correctly
# </REASONING_QUALITY>

# # INPUT PROCESSING
# You will receive:
# - **Stage-Specific Instructions**: Detailed task description with mathematical context
# - **Context Data**: Relevant input data formatted for the specific algorithmic step
# - **Output Schema**: Exact JSON structure specification for your response

# # OUTPUT REQUIREMENTS

# ## <CRITICAL_CONSTRAINTS>
# - **JSON Only**: Return EXCLUSIVELY valid JSON, no additional text, markdown, or explanations
# - **Schema Compliance**: Every output key must match the provided schema exactly
# - **Mathematical Accuracy**: Ensure all calculations and transformations are correct
# - **Complete Coverage**: Address all aspects of the stage instructions thoroughly
# </CRITICAL_CONSTRAINTS>

# ## <QUALITY_STANDARDS>
# - **Precision**: Mathematical operations must be exact and well-reasoned
# - **Completeness**: All required outputs must be fully populated
# - **Consistency**: Results must be internally coherent and logically sound
# - **Algorithmic Correctness**: Apply the specified algorithm principles accurately

# # EXECUTION APPROACH
# 1. **Parse Instructions**: Understand the specific mathematical/algorithmic task
# 2. **Analyze Context**: Process all provided input data systematically
# 3. **Apply Methodology**: Execute the required algorithmic steps with precision
# 4. **Validate Results**: Ensure outputs meet mathematical and structural requirements
# 5. **Format Response**: Return perfectly compliant JSON matching the schema

# **CRITICAL SUCCESS FACTOR**: Your output must be both mathematically correct AND perfectly formatted according to the specified schema. No compromises on either dimension.
# """

    executor = Agent("openai:o3-mini", output_type=str, system_prompt=system_prompt)

    return executor


def preprocess_template(template: str, read_data: Dict[str, Any]) -> str:
    """Preprocess template to avoid input repetition by replacing subsequent placeholder references"""
    import re

    # Get valid keys from read_data
    valid_keys = set(read_data.keys())

    # First, clean up invalid placeholders (those not in read_data keys)
    def clean_invalid_placeholders(match):
        placeholder = match.group(1)
        if placeholder in valid_keys:
            return match.group(0)  # Keep valid placeholders intact
        else:
            return placeholder  # Remove braces from invalid placeholders

    # Remove braces from invalid placeholders only
    template = re.sub(r'\{(\w+)\}', clean_invalid_placeholders, template)

    # Then, process valid placeholders to avoid repetition
    for key in valid_keys:
        pattern = f'{{{key}}}'

        # Find all occurrences of this specific read_data key
        occurrences = []
        for match in re.finditer(re.escape(pattern), template):
            occurrences.append((match.start(), match.end()))

        if len(occurrences) > 1:
            # Find the INPUT DATA section
            input_data_pos = template.find('# INPUT DATA')

            if input_data_pos >= 0:
                # Strategy: Keep ONLY the first occurrence in INPUT DATA section, replace ALL others
                first_in_input_data = None
                replacements = []

                for start, end in occurrences:
                    if start > input_data_pos:
                        # This is after INPUT DATA section
                        if first_in_input_data is None:
                            first_in_input_data = (start, end)  # Keep this one
                        else:
                            # Replace this occurrence
                            replacements.append((start, end, f'the data provided above'))
                    else:
                        # This is before INPUT DATA section (e.g., in TASK), replace it
                        replacements.append((start, end, f'the input data'))

                # Apply replacements in reverse order to maintain positions
                for start, end, replacement in reversed(replacements):
                    template = template[:start] + replacement + template[end:]

            else:
                # No INPUT DATA section found, replace all but the first occurrence
                replacements = []
                for i, (start, end) in enumerate(occurrences[1:], 1):
                    replacements.append((start, end, f'the input data'))

                # Apply replacements in reverse order to maintain positions
                for start, end, replacement in reversed(replacements):
                    template = template[:start] + replacement + template[end:]

    return template


async def run_stage(stage: Stage, context: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Execute a single stage of the plan.

    Args:
        stage: The stage to execute
        context: Current execution context
        verbose: If True, print execution details; if False, suppress output
    """
    import time

    logger = get_logger()
    executor = create_executor()

    # Get the data this stage needs to read
    read_data = {key: context.get(key) for key in stage.reads}

    if verbose:
        logger.stage_header(stage.id, stage.reads, stage.writes)

    # logger.debug_print(f"\n🔍 DEBUG - Stage {stage.id}:")
    # logger.debug_print(f"     📥 Input data: {read_data}")
    # logger.debug_print(f"     📋 Stage prompt template: {stage.prompt_template[:200]}...")

    start_time = time.time()

    # Render the prompt template with the read data
    try:
        # Preprocess template to avoid input repetition
        processed_template = preprocess_template(stage.prompt_template, read_data)

        # First, escape any braces that aren't actual placeholders
        escaped_template = processed_template

        # Find all placeholders that should be replaced (those that match keys in read_data)
        import re

        actual_placeholders = set(read_data.keys())

        # Replace all {text} that are NOT actual placeholders with {{text}}
        def escape_non_placeholders(match):
            placeholder = match.group(1)
            # Handle both simple keys and complex content in braces
            if placeholder in actual_placeholders:
                return match.group(0)  # Keep as {placeholder}
            else:
                # Escape any brace content that's not a valid placeholder
                return "{{" + placeholder + "}}"  # Escape as {{placeholder}}

        # More robust pattern to catch all brace patterns
        escaped_template = re.sub(
            r"\{([^}]+)\}", escape_non_placeholders, escaped_template
        )

        # Additional safety: Use string.Template as fallback for complex cases
        from string import Template

        # Try format first, fallback to Template if it fails
        try:
            rendered_prompt = escaped_template.format(**read_data)
        except (IndexError, KeyError, ValueError):
            # Fallback: Convert to Template syntax and substitute
            # Convert {key} to $key for Template
            template_str = escaped_template
            for key in actual_placeholders:
                template_str = template_str.replace("{" + key + "}", "$" + key)

            template = Template(template_str)
            rendered_prompt = template.safe_substitute(**read_data)
    except KeyError as e:
        raise ValueError(
            f"Stage '{stage.id}' tried to read key {e} that doesn't exist in context"
        )

    # Validate that all placeholders were resolved
    import re
    unresolved_placeholders = re.findall(r'\{(\w+)\}', rendered_prompt)
    if unresolved_placeholders:
        raise ValueError(
            f"Stage '{stage.id}' has unresolved placeholders: {unresolved_placeholders}. "
            f"This indicates missing data or template issues. "
            f"Available read data keys: {list(read_data.keys())}"
        )

    # Add schema information to guide the output
    prompt_with_schema = f"{rendered_prompt}\n\nOutput JSON Schema:\n{json.dumps(stage.output_schema, indent=2)}"

    # logger.debug_print(f"     📝 Rendered prompt (first 500 chars): {prompt_with_schema[:500]}...")
    # if len(prompt_with_schema) > 500:
    #     logger.debug_print(f"     📝 Rendered prompt (last 200 chars): ...{prompt_with_schema[-200:]}")

    # Log RAW INPUT (only in debug mode)
    logger.model_call_separator(stage.id)
    logger.raw_input(prompt_with_schema)

    # Execute the stage
    result = await executor.run(prompt_with_schema)
    json_output = result.output

    # Log RAW OUTPUT (only in debug mode)
    logger.raw_output(json_output)

    # Parse the JSON response
    try:
        stage_output = json.loads(json_output)
        # if not logger.debug:
        #     print("✅", end="")
        # logger.debug_print("✅")
        # logger.debug_print(f"     ✅ JSON parsing successful")
    except json.JSONDecodeError:
        try:
            stage_output = ast.literal_eval(json_output)
            if not logger.debug:
                print("✅", end="")
            logger.debug_print("✅")
            logger.debug_print(f"     ✅ Fallback parsing successful")
        except (ValueError, SyntaxError) as e:
            print("❌")
            logger.debug_print(f"     ❌ JSON/dict parsing failed: {e}")
            raise ValueError(f"Stage '{stage.id}' returned invalid JSON/dict: {e}")

    # Validate that all required keys are present
    missing_keys = [key for key in stage.writes if key not in stage_output]
    if missing_keys:
        print(f"❌ Missing: {missing_keys}")
        logger.debug_print(f"     ❌ DEBUG - Missing keys: {missing_keys}")
        logger.debug_print(f"     ❌ DEBUG - Expected keys: {stage.writes}")
        logger.debug_print(f"     ❌ DEBUG - Actual keys: {list(stage_output.keys())}")
        raise ValueError(
            f"Stage '{stage.id}' did not produce required output keys: {missing_keys}"
        )

    # logger.debug_print(f"     ✅ All required output keys present: {stage.writes}")

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
                output_preview.append(
                    f"{key}:[{', '.join(str(k) for k in preview_keys)}]"
                )
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

    # In normal mode: show concise completion info
    if verbose and not logger.debug:
        logger.stage_complete(execution_time, ', '.join(output_info))

        # Show output preview on next line with indentation
        if output_preview:
            logger.stage_output_preview(', '.join(output_preview))

        # Show compact summary (prompt + input + output)
        # Extract first line of prompt as preview
        prompt_lines = stage.prompt_template.split('\n')
        # Find the TASK section
        task_line = None
        for i, line in enumerate(prompt_lines):
            if line.strip().startswith('# TASK'):
                if i + 1 < len(prompt_lines):
                    task_line = prompt_lines[i + 1].strip()
                    break
        prompt_preview = task_line if task_line else prompt_lines[0][:100]

        logger.stage_summary(prompt_preview, read_data, stage_output)

    # In debug mode: show full output details
    else:
        # logger.stage_complete(execution_time, ', '.join(output_info))
        # if output_preview:
        #     logger.stage_output_preview(', '.join(output_preview))
        # logger.full_output(stage_output)
        test = 2

    return stage_output


async def run_plan(plan: Plan, initial_context: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Execute a complete plan by running all stages sequentially.

    Args:
        plan: The execution plan with stages
        initial_context: Initial context data
        verbose: If True, print detailed execution logs; if False, suppress output
    """
    import time

    logger = get_logger()
    plan_start_time = time.time()

    if verbose:
        print(f"\n🎯 Executing {len(plan.stages)} stages:")
    context = dict(initial_context)

    # logger.debug_print(f"🔍 DEBUG - Initial context: {initial_context}")
    # logger.debug_print(f"🔍 DEBUG - Plan stages: {[stage.id for stage in plan.stages]}")

    # Run each stage sequentially
    for i, stage in enumerate(plan.stages, 1):
        try:
            # Execute the stage
            stage_output = await run_stage(stage, context, verbose=verbose)

            # Update context with stage outputs
            for key in stage.writes:
                context[key] = stage_output[key]

            # logger.debug_print(f"🔍 DEBUG - Context after {stage.id}: {list(context.keys())}")

        except Exception as e:
            print(f"❌ ERROR in {stage.id}: {e}")
            raise

    total_time = time.time() - plan_start_time
    final_result = context.get(plan.final_key, "<<NOT_FOUND>>")
    if verbose:
        print(f"\n🎯 Completed in {total_time:.1f}s → {final_result}")

    return context
