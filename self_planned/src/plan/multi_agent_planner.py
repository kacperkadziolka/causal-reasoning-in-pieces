"""
Multi-Agent Planning Pipeline

This module implements a decomposed planning approach that separates concerns:
1. Stage Sequence Generator: Determines high-level algorithmic stages
2. Prompt Designer: Creates detailed prompts for each stage
3. Schema Designer: Generates precise schemas for each stage
4. Plan Validator & Aligner: Ensures consistency and fixes issues

Benefits:
- Reduced cognitive load per agent (simpler, focused tasks)
- Better error isolation and debugging
- More reliable outputs through specialization
- Easier validation at each step
"""

from pydantic_ai import Agent
from typing import Dict, Any, List, Tuple
import asyncio
import json
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from models import Plan, Stage
from utils.logging_config import get_logger


# Intermediate data models for the pipeline
class StageSequence(BaseModel):
    """High-level stage sequence without implementation details"""
    stages: List[Dict[str, Any]]  # Each: {id, purpose, reads, writes}


class PromptLibrary(BaseModel):
    """Collection of detailed prompts for each stage"""
    prompts: Dict[str, str]  # stage_id -> prompt_template


class SchemaLibrary(BaseModel):
    """Collection of schemas for each stage"""
    schemas: Dict[str, Dict[str, Any]]  # stage_id -> output_schema


class MultiAgentPlanner:
    """
    Multi-agent planning pipeline with separation of concerns.

    Architecture:
    1. Sequence Agent: Algorithm analysis → stage sequence
    2. Prompt Agent: Stage purpose → detailed prompt templates
    3. Schema Agent: Stage outputs → precise JSON schemas
    4. Validator Agent: Alignment and consistency checking
    """

    def __init__(self):
        # Agent 1: Stage Sequence Generator
        self.sequence_agent = Agent(
            "openai:o3-mini",
            output_type=StageSequence,
            system_prompt="""
# ROLE
You are an algorithm decomposition specialist.

# TASK
Analyze an algorithm and decompose it into a sequence of high-level stages.

# FOCUS
- What are the main algorithmic phases?
- What mathematical objects are created/consumed at each phase?
- What is the data flow between phases?

# OUTPUT FORMAT
Return a StageSequence with stages array containing:
- id: descriptive stage identifier (snake_case)
- purpose: what this stage accomplishes mathematically
- reads: context keys this stage needs (from prior stages or initial input)
- writes: context keys this stage produces

# RULES
1. Each stage represents ONE algorithmic phase/transformation
2. reads must be available from initial context or prior stage writes
3. writes must be unique per stage (no overwrites without good reason)
4. Keep stages at a high conceptual level (details come later)
5. Ensure proper data flow connectivity

# EXAMPLE
For "Peter-Clark (PC) Algorithm":
```json
{
  "stages": [
    {
      "id": "skeleton_construction",
      "purpose": "Build initial undirected graph from correlations",
      "reads": ["input"],
      "writes": ["skeleton"]
    },
    {
      "id": "sepset_recording",
      "purpose": "Record separation sets for removed edges",
      "reads": ["input", "skeleton"],
      "writes": ["sepsets"]
    },
    ...
  ]
}
```

Focus on algorithmic correctness and data flow, NOT implementation details.
"""
        )

        # Agent 2: Prompt Designer
        self.prompt_agent = Agent(
            "openai:o3-mini",
            output_type=PromptLibrary,
            system_prompt="""
# ROLE
You are a prompt engineering specialist for LLM execution stages.

# TASK
Given high-level stage descriptions, create detailed prompt templates that will guide LLM execution.

# INPUT
You receive:
- Algorithm knowledge (mathematical foundations)
- Stage sequence (id, purpose, reads, writes)

# OUTPUT FORMAT
Return PromptLibrary with prompts dictionary:
- Keys: stage IDs
- Values: detailed prompt templates

# PROMPT TEMPLATE STRUCTURE
EVERY prompt template MUST follow this exact format:

```
# TASK
[One clear sentence describing what to do - NO placeholders here]

# INPUT DATA
{read_key_1}
{read_key_2}
[... one {placeholder} line for EACH key in the stage's reads array]

# STEP-BY-STEP
1. [Instruction referencing "the data provided above" - NO placeholders]
2. [Instruction referencing "the input" - NO placeholders]
3. [Instruction referencing specific aspects - NO placeholders]

# OUTPUT
Return JSON with the key(s): [list write keys for this stage]
```

# CRITICAL RULES
1. INPUT DATA section: EXACTLY one {placeholder} for each reads key
2. TASK section: NO placeholders, just description
3. STEP-BY-STEP: NO placeholders, use references like "the data above"
4. OUTPUT section: Specify exact keys that match writes array
5. Keep prompts focused on DATA TRANSFORMATION, not algorithmic theory
6. Use concrete, actionable language

# QUALITY CRITERIA
- Clear, unambiguous instructions
- Focuses on transforming input → output
- Includes validation hints where appropriate
- References mathematical context from algorithm knowledge
- Each step is a specific action on the input data

Generate comprehensive, execution-ready prompts.
"""
        )

        # Agent 3: Schema Designer
        self.schema_agent = Agent(
            "openai:o3-mini",
            output_type=SchemaLibrary,
            system_prompt="""
# ROLE
You are a JSON schema specialist for mathematical data structures.

# TASK
Generate precise JSON schemas for each stage's output.

# INPUT
You receive:
- Stage sequence (id, purpose, writes)
- Algorithm knowledge (mathematical objects and their formats)

# OUTPUT FORMAT
Return SchemaLibrary with schemas dictionary:
- Keys: stage IDs
- Values: JSON Schema objects (valid JSON Schema format)

# SCHEMA STRUCTURE
Each schema MUST follow this pattern:

```json
{
  "type": "object",
  "properties": {
    "write_key_1": { /* detailed schema */ },
    "write_key_2": { /* detailed schema */ }
  },
  "required": ["write_key_1", "write_key_2"]
}
```

# CRITICAL RULES
1. properties keys MUST exactly match the stage's writes array
2. required array MUST exactly match the stage's writes array
3. Use specific types when possible (not just "object")
4. For mathematical structures, use detailed schemas:
   - Graphs: specify nodes, edges structure
   - Sets: specify array with item constraints
   - Mappings: specify object with patternProperties
5. Include descriptions for clarity

# EXAMPLE
For writes: ["graph", "sepsets"]
```json
{
  "type": "object",
  "properties": {
    "graph": {
      "type": "object",
      "properties": {
        "nodes": {"type": "array", "items": {"type": "string"}},
        "edges": {"type": "array", "items": {
          "type": "object",
          "properties": {
            "from": {"type": "string"},
            "to": {"type": "string"},
            "directed": {"type": "boolean"}
          }
        }}
      }
    },
    "sepsets": {
      "type": "object",
      "patternProperties": {
        "^[A-Z]-[A-Z]$": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    }
  },
  "required": ["graph", "sepsets"]
}
```

# QUALITY CRITERIA
- Precise type definitions
- Complete property specifications
- Matches mathematical object structure from algorithm knowledge
- Executable by LLM (clear, unambiguous)

# CRITICAL: FINAL STAGE SCHEMA VALIDATION
When generating schemas for the FINAL stage of the plan:
1. Identify if this is the final stage (last stage in sequence)
2. For write keys representing boolean decisions:
   - USE simple boolean type: {"type": "boolean"}
   - DO NOT use nested objects with properties

Example CORRECT: {"decision": {"type": "boolean"}}
Example INCORRECT: {"decision": {"type": "object", "properties": {"verified": {...}}}}

Generate precise, mathematically-sound schemas.
"""
        )

        # Agent 4: Plan Validator & Aligner
        self.validator_agent = Agent(
            "openai:o3-mini",
            output_type=Plan,
            system_prompt="""
# ROLE
You are a plan consistency validator and alignment specialist.

# TASK
Given stage sequence, prompts, and schemas, assemble a complete Plan and ensure perfect consistency.

# INPUT
You receive:
- Stage sequence (id, purpose, reads, writes)
- Prompt library (stage_id → prompt_template)
- Schema library (stage_id → output_schema)
- Final key designation

# OUTPUT
Return a complete Plan object with stages array and final_key.

# VALIDATION & ALIGNMENT
For each stage, ensure:

1. **Placeholder Alignment**
   - Extract reads keys from sequence
   - Verify prompt has {placeholder} for EACH reads key
   - Fix if missing: add to INPUT DATA section

2. **Schema-Writes Alignment**
   - Extract writes keys from sequence
   - Verify schema.properties has EXACT match to writes
   - Verify schema.required matches writes
   - Fix if misaligned: adjust schema to match writes (canonical)

3. **Data Flow Connectivity**
   - Verify each reads key is available (from initial context or prior writes)
   - Flag any broken dependencies

4. **Output Consistency**
   - Verify OUTPUT section in prompt mentions correct keys
   - Update if needed to match writes

# AUTO-CORRECTION RULES
- writes array is CANONICAL for what a stage produces
- If schema doesn't match writes: adjust schema to match
- If prompt placeholders don't match reads: add missing placeholders
- Report all corrections made

# ASSEMBLY
Create complete Stage objects with:
- id, reads, writes (from sequence)
- prompt_template (from prompt library, validated)
- output_schema (from schema library, validated)

# QUALITY ASSURANCE
- No mismatches between writes and schema properties
- All placeholders present for reads keys
- Proper data flow connectivity
- Clear final_key designation

# CRITICAL: FINAL STAGE VALIDATION AND ENFORCEMENT
For the LAST stage in the plan:

1. **Mandatory Schema Check**:
   - For each write key: schema.properties[key]["type"] MUST be "boolean"
   - NOT "object" with nested properties

2. **Auto-Correction Procedure**:
   - IF nested object detected in final stage schema
   - THEN simplify to: {"type": "boolean"}

   Example transformation:
   BEFORE: {"decision": {"type": "object", "properties": {"holds": {"type": "boolean"}}}}
   AFTER:  {"decision": {"type": "boolean"}}

3. **Validation Check**:
   - Verify: final_stage.output_schema.properties[final_key]["type"] == "boolean"
   - If not, CRITICAL ERROR - must fix before returning plan

Return the validated, aligned, complete Plan.
"""
        )

        # Agent 5: Sequential Stage Detail Generator (for sequential mode)
        # Define output model for structured response
        class StageDetails(BaseModel):
            prompt_template: str
            output_schema: Dict[str, Any]

        self.sequential_stage_agent = Agent(
            "openai:o3-mini",
            output_type=StageDetails,
            system_prompt="""
# ROLE
You are an expert at creating detailed, executable stage specifications.

# TASK
Generate a DETAILED prompt template and JSON schema for ONE specific stage in an algorithm execution plan.

# CONTEXT YOU RECEIVE
- Algorithm knowledge (mathematical foundations)
- Current stage specification (ID, purpose, reads, writes)
- All previously generated stages (with their prompts and schemas)
- Available data from previous stages
- Position in the pipeline (stage N of M)

# OUTPUT REQUIREMENTS
Return a StageDetails object with:
- prompt_template: detailed prompt with placeholders
- output_schema: JSON schema object

# PROMPT TEMPLATE STRUCTURE
Your prompt_template MUST follow this exact format:

```
# TASK
[One clear sentence describing what to do - NO placeholders here]

# CONTEXT FROM PREVIOUS STAGES
[Explicitly reference relevant outputs from prior stages]
[Example: "Using the skeleton graph from stage 1..." or "Building on the separation sets identified in stage 2..."]
[SKIP this section if this is the first stage]

# INPUT DATA
{read_key_1}
{read_key_2}
[... one {placeholder} line for EACH key in the stage's reads array]

Available data at this point: {available_data}

# STEP-BY-STEP INSTRUCTIONS
1. [Detailed step using specific inputs - reference previous stage outputs when relevant]
2. [Build on previous stage outputs explicitly if applicable]
3. [More detailed instructions...]

# OUTPUT
Return JSON with the key(s): [list write keys for this stage]
```

# CRITICAL RULES FOR PROMPTS
1. **Context References**: Explicitly mention how this stage builds on previous stages (if applicable)
2. **Input Data Section**: EXACTLY one {placeholder} for each reads key
3. **Task Section**: NO placeholders, just clear description
4. **Step-by-Step**: NO placeholders in instructions, use references like "the data above" or "the skeleton from earlier"
5. **Detail Level**: Be 2-3× more detailed than a batch-generated prompt
6. **Data Flow**: Make the connection to previous stages explicit and clear
7. **Placeholders**: All placeholders must exist in {available_data}

# SCHEMA STRUCTURE
Your output_schema MUST follow this pattern:

```json
{
  "type": "object",
  "properties": {
    "write_key_1": { /* detailed schema */ },
    "write_key_2": { /* detailed schema */ }
  },
  "required": ["write_key_1", "write_key_2"]
}
```

# CRITICAL RULES FOR SCHEMAS
1. properties keys MUST exactly match the stage's writes array
2. required array MUST exactly match the stage's writes array
3. Use specific types when possible (not just "object")
4. For mathematical structures, use detailed schemas:
   - Graphs: specify nodes, edges structure
   - Sets: specify array with item constraints
   - Mappings: specify object with patternProperties
5. Include descriptions for clarity
6. Consider compatibility with downstream stages
7. **Final Stage Boolean Schema**:
   - IF position indicates last stage (e.g., "stage 6 of 6")
   - AND write key represents a decision/result
   - THEN use simple boolean schema: {"type": "boolean"}
   - DO NOT use nested objects for final boolean outputs

# QUALITY CRITERIA
- Prompt is significantly more detailed than batch generation
- Explicit references to previous stage outputs (where applicable)
- Clear data flow context
- Precise, mathematically-sound schema
- All placeholders are valid (in available_data)
- Instructions are concrete and actionable

Generate the most detailed, robust prompt and schema possible for this ONE stage.
"""
        )

    async def generate_plan(
        self,
        task_description: str,
        algorithm_knowledge: str,
        use_sequential: bool = False,
        max_retries: int = 2
    ) -> Tuple[Plan, Dict[str, Any]]:
        """
        Generate a plan using multi-agent pipeline.

        Args:
            task_description: Task description template
            algorithm_knowledge: Extracted algorithm knowledge
            use_sequential: If True, generate stage prompts sequentially
                           (one LLM call per stage with feed-forward context).
                           If False, generate all prompts in one batch call.
                           Sequential mode produces more detailed prompts but
                           requires ~2× more LLM calls.
            max_retries: Maximum validation retry attempts

        Returns:
            (plan, metadata) where metadata contains intermediate results for debugging
        """
        logger = get_logger()

        logger.section("🎯 MULTI-AGENT PLANNING PIPELINE")

        metadata = {}

        # Step 1: Generate Stage Sequence
        logger.subsection("📋 STEP 1: Generating Stage Sequence...")
        sequence = await self._generate_sequence(task_description, algorithm_knowledge)
        metadata["sequence"] = sequence.model_dump()
        logger.success(f"Generated {len(sequence.stages)} stages")
        for stage in sequence.stages:
            logger.planning_progress(f"   - {stage['id']}: {stage['reads']} → {stage['writes']}", show_always=True)

        # Step 2 & 3: Generate Prompts and Schemas
        if use_sequential:
            logger.subsection("✍️  STEP 2 & 3: Designing Prompts & Schemas SEQUENTIALLY (one stage at a time)...")
            prompt_library, schema_library = await self._generate_prompts_and_schemas_sequential(
                stage_sequence=sequence.stages,
                algorithm_knowledge=algorithm_knowledge
            )
            # Convert to expected format
            prompts = PromptLibrary(prompts=prompt_library)
            schemas = SchemaLibrary(schemas=schema_library)
            metadata["prompts"] = prompts.model_dump()
            metadata["schemas"] = schemas.model_dump()
            metadata["generation_mode"] = "sequential"
            logger.success(f"Generated {len(prompts.prompts)} detailed prompt templates (sequential mode)")
        else:
            logger.subsection("✍️  STEP 2: Designing Prompt Templates (BATCH mode)...")
            prompts = await self._generate_prompts(sequence, algorithm_knowledge)
            metadata["prompts"] = prompts.model_dump()
            logger.success(f"Generated {len(prompts.prompts)} prompt templates")

            logger.subsection("📐 STEP 3: Designing Output Schemas (BATCH mode)...")
            schemas = await self._generate_schemas(sequence, algorithm_knowledge)
            metadata["schemas"] = schemas.model_dump()
            metadata["generation_mode"] = "batch"
            logger.success(f"Generated {len(schemas.schemas)} schemas")

        # Step 4: Validate & Align
        logger.subsection("🔍 STEP 4: Validating & Aligning Plan...")

        # Determine final key (last stage's first write, or explicit from task)
        final_key = self._determine_final_key(sequence, task_description)

        plan = await self._validate_and_align(
            sequence, prompts, schemas, final_key, max_retries
        )
        metadata["final_key"] = final_key
        logger.success("Plan validated and aligned")
        logger.info(f"   Final output key: '{final_key}'")

        logger.section("✅ MULTI-AGENT PLANNING COMPLETE")

        return plan, metadata

    async def _generate_sequence(
        self, task_description: str, algorithm_knowledge: str
    ) -> StageSequence:
        """Step 1: Generate high-level stage sequence"""

        prompt = f"""
Analyze this algorithm and task to generate a stage sequence.

ALGORITHM KNOWLEDGE:
{algorithm_knowledge}

TASK:
{task_description}

Generate a logical sequence of stages that implements this algorithm for the given task.
Each stage should represent one algorithmic phase with clear inputs and outputs.
"""

        result = await self.sequence_agent.run(prompt)
        return result.output

    async def _generate_prompts(
        self, sequence: StageSequence, algorithm_knowledge: str
    ) -> PromptLibrary:
        """Step 2: Generate detailed prompt templates"""

        prompt = f"""
Generate detailed prompt templates for each stage in this sequence.

ALGORITHM KNOWLEDGE:
{algorithm_knowledge}

STAGE SEQUENCE:
{json.dumps(sequence.model_dump(), indent=2)}

For each stage, create a comprehensive prompt template that will guide LLM execution.
Remember: INPUT DATA section must have EXACTLY one {{placeholder}} for each reads key.
"""

        result = await self.prompt_agent.run(prompt)
        return result.output

    async def _generate_schemas(
        self, sequence: StageSequence, algorithm_knowledge: str
    ) -> SchemaLibrary:
        """Step 3: Generate precise schemas"""

        prompt = f"""
Generate precise JSON schemas for each stage's outputs.

ALGORITHM KNOWLEDGE:
{algorithm_knowledge}

STAGE SEQUENCE:
{json.dumps(sequence.model_dump(), indent=2)}

For each stage, create a detailed schema that exactly matches the writes array.
Use specific mathematical structures from the algorithm knowledge where applicable.
"""

        result = await self.schema_agent.run(prompt)
        return result.output

    async def _validate_and_align(
        self,
        sequence: StageSequence,
        prompts: PromptLibrary,
        schemas: SchemaLibrary,
        final_key: str,
        max_retries: int
    ) -> Plan:
        """Step 4: Validate and align all components into a complete Plan"""

        for attempt in range(max_retries):
            prompt = f"""
Assemble and validate a complete Plan from these components.

STAGE SEQUENCE:
{json.dumps(sequence.model_dump(), indent=2)}

PROMPT LIBRARY:
{json.dumps(prompts.model_dump(), indent=2)}

SCHEMA LIBRARY:
{json.dumps(schemas.model_dump(), indent=2)}

FINAL KEY: {final_key}

Validate consistency and align all components. Auto-correct any mismatches.
Return a complete, validated Plan.
"""

            result = await self.validator_agent.run(prompt)
            plan = result.output

            # Additional validation (our Python-side checks)
            validation_errors = self._validate_plan_structure(plan)

            if not validation_errors:
                logger = get_logger()
                logger.success(f"Validation passed on attempt {attempt + 1}")
                return plan

            logger = get_logger()
            logger.warning(f"Validation issues found on attempt {attempt + 1}:")
            for error in validation_errors:
                logger.info(f"   - {error}")

            if attempt < max_retries - 1:
                logger.info("🔄 Retrying validation...")
                # Could add error feedback to next validation attempt
            else:
                logger.warning("Max retries reached, returning plan with warnings")
                return plan

        return plan

    def _determine_final_key(self, sequence: StageSequence, task_description: str) -> str:
        """Determine which key contains the final result"""
        # Heuristic: last stage's first write key
        # Could be enhanced with LLM analysis of task_description
        if sequence.stages:
            last_stage = sequence.stages[-1]
            if last_stage.get('writes'):
                return last_stage['writes'][0]

        return "result"  # fallback

    def _validate_plan_structure(self, plan: Plan) -> List[str]:
        """
        Validate plan structure and return list of errors.
        Returns empty list if valid.
        """
        import re
        errors = []

        for stage in plan.stages:
            # Check schema-writes alignment
            schema_keys = set(stage.output_schema.get('properties', {}).keys())
            writes_keys = set(stage.writes)

            if schema_keys != writes_keys:
                errors.append(
                    f"Stage '{stage.id}': schema keys {schema_keys} != writes {writes_keys}"
                )

            # Check placeholder presence
            placeholders = set(re.findall(r'\{(\w+)\}', stage.prompt_template))
            reads_keys = set(stage.reads)
            missing = reads_keys - placeholders

            if missing:
                errors.append(
                    f"Stage '{stage.id}': missing placeholders for {missing}"
                )

        # Validate final stage uses simple boolean schema
        if plan.stages:
            final_stage = plan.stages[-1]
            final_key = plan.final_key or (final_stage.writes[0] if final_stage.writes else None)

            if final_key and final_key in final_stage.output_schema.get('properties', {}):
                final_schema = final_stage.output_schema['properties'][final_key]

                # Check for nested object (forbidden)
                is_nested_object = (
                    final_schema.get('type') == 'object' and
                    'properties' in final_schema
                )

                # Check for simple boolean (required)
                is_simple_boolean = final_schema.get('type') == 'boolean'

                if is_nested_object:
                    errors.append(
                        f"Final stage '{final_stage.id}': key '{final_key}' uses nested object schema. "
                        f"MUST be simple boolean: {{'type': 'boolean'}}. Found: {final_schema}"
                    )
                elif not is_simple_boolean:
                    errors.append(
                        f"Final stage '{final_stage.id}': key '{final_key}' must use boolean type. "
                        f"Found: {final_schema.get('type')}"
                    )

        return errors

    # Sequential generation methods (Phase 2)

    def _compute_available_data(self, previous_stages: List[Dict]) -> List[str]:
        """Compute what data is available for the current stage."""
        available = ["input"]  # Always start with input
        for stage in previous_stages:
            available.extend(stage.get("writes", []))
        return available

    async def _generate_stage_details_sequential(
        self,
        stage_spec: Dict,
        algorithm_knowledge: str,
        previous_stages: List[Dict],
        position: str
    ) -> Dict:
        """
        Generate detailed prompt + schema for ONE stage.

        Args:
            stage_spec: Stage from sequence (id, purpose, reads, writes)
            algorithm_knowledge: Algorithm knowledge text
            previous_stages: List of already-generated stages with prompts/schemas
            position: "stage 3 of 6" for context

        Returns:
            Dict with prompt_template and output_schema
        """
        available_data = self._compute_available_data(previous_stages)

        user_message = f"""Generate detailed prompt and schema for this stage:

CURRENT STAGE:
- ID: {stage_spec['id']}
- Purpose: {stage_spec['purpose']}
- Reads: {stage_spec['reads']}
- Writes: {stage_spec['writes']}
- Position: {position}

ALGORITHM KNOWLEDGE:
{algorithm_knowledge}

PREVIOUS STAGES:
{json.dumps(previous_stages, indent=2) if previous_stages else "None (this is the first stage)"}

AVAILABLE DATA:
{available_data}

Generate a prompt template that explicitly builds on previous stages and a schema that matches the writes array."""

        try:
            result = await self.sequential_stage_agent.run(user_message)
            # Extract from structured output (Pydantic model)
            stage_details = result.output
            return {
                "prompt_template": stage_details.prompt_template,
                "output_schema": stage_details.output_schema
            }
        except Exception as e:
            logger = get_logger()
            logger.error(f"Sequential stage generation failed for {stage_spec['id']}: {e}")
            if logger.debug:
                import traceback
                traceback.print_exc()
            raise

    async def _generate_prompts_and_schemas_sequential(
        self,
        stage_sequence: List[Dict],
        algorithm_knowledge: str
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """
        Generate prompts and schemas sequentially (one stage at a time).

        Returns:
            (prompt_library, schema_library)
        """
        prompt_library = {}
        schema_library = {}
        generated_stages = []

        total_stages = len(stage_sequence)

        logger = get_logger()

        for idx, stage_spec in enumerate(stage_sequence):
            stage_id = stage_spec['id']
            position = f"stage {idx + 1} of {total_stages}"

            logger.planning_progress(f"   Generating details for {stage_id} ({position})...", show_always=True)

            # Generate prompt + schema with context of all previous stages
            details = await self._generate_stage_details_sequential(
                stage_spec=stage_spec,
                algorithm_knowledge=algorithm_knowledge,
                previous_stages=generated_stages,
                position=position
            )

            # Store in libraries
            prompt_library[stage_id] = details["prompt_template"]
            schema_library[stage_id] = details["output_schema"]

            # Add to generated stages for next iteration
            generated_stages.append({
                "id": stage_id,
                "purpose": stage_spec['purpose'],
                "reads": stage_spec['reads'],
                "writes": stage_spec['writes'],
                "prompt_template": details["prompt_template"],
                "output_schema": details["output_schema"]
            })

            logger.planning_progress(f"   ✓ Generated {stage_id}: prompt={len(details['prompt_template'])} chars", show_always=True)

        return prompt_library, schema_library


# Test function
async def test_multi_agent_planning():
    """Test the multi-agent planning pipeline"""

    planner = MultiAgentPlanner()

    task_description = """
Task: Given a natural-language input containing a Premise and a Hypothesis, decide whether the Hypothesis is True or False under the Peter-Clark (PC) algorithm.

Input available in context: 'input' (contains premise with variables, correlations, conditional independencies, and hypothesis).

CRITICAL OUTPUT FORMAT: The final stage must output ONLY a boolean value (true or false).
"""

    algorithm_knowledge = """
## <DEFINITION>
The Peter-Clark (PC) algorithm is a constraint-based causal discovery method that infers causal structure from conditional independence information.
</DEFINITION>

## <CANONICAL_STAGES>
1. **Skeleton Construction**: Build initial undirected graph by connecting correlated variables
2. **Edge Removal**: Remove edges based on conditional independence tests
3. **Separation Set Recording**: Record conditioning sets that made variables independent
4. **V-Structure Identification**: Find colliders using separation sets
5. **Edge Orientation**: Apply Meek's rules to orient remaining edges
6. **Hypothesis Evaluation**: Test hypothesis against final CPDAG
</CANONICAL_STAGES>

## <KEY_MATHEMATICAL_OBJECTS>
- Graph: nodes (variables) and edges (connections)
- Separation Sets: mapping of variable pairs to conditioning sets
- CPDAG: completed partially directed acyclic graph
</KEY_MATHEMATICAL_OBJECTS>
"""

    print("🧪 Testing Multi-Agent Planning Pipeline")
    print("=" * 60)

    try:
        plan, metadata = await planner.generate_plan(
            task_description=task_description,
            algorithm_knowledge=algorithm_knowledge
        )

        print("\n📊 RESULTS:")
        print(f"Number of stages: {len(plan.stages)}")
        print(f"Final key: {plan.final_key}")

        print("\n📋 Stage Summary:")
        for i, stage in enumerate(plan.stages, 1):
            print(f"{i}. {stage.id}")
            print(f"   Reads:  {stage.reads}")
            print(f"   Writes: {stage.writes}")

        return plan, metadata

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, {}


if __name__ == "__main__":
    asyncio.run(test_multi_agent_planning())
