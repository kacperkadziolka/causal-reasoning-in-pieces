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

Return the validated, aligned, complete Plan.
"""
        )

    async def generate_plan(
        self,
        task_description: str,
        algorithm_knowledge: str,
        max_retries: int = 2
    ) -> Tuple[Plan, Dict[str, Any]]:
        """
        Generate a plan using multi-agent pipeline.

        Returns:
            (plan, metadata) where metadata contains intermediate results for debugging
        """
        print("\n🎯 MULTI-AGENT PLANNING PIPELINE")
        print("=" * 60)

        metadata = {}

        # Step 1: Generate Stage Sequence
        print("\n📋 STEP 1: Generating Stage Sequence...")
        sequence = await self._generate_sequence(task_description, algorithm_knowledge)
        metadata["sequence"] = sequence.model_dump()
        print(f"✅ Generated {len(sequence.stages)} stages")
        for stage in sequence.stages:
            print(f"   - {stage['id']}: {stage['reads']} → {stage['writes']}")

        # Step 2: Generate Prompts
        print("\n✍️  STEP 2: Designing Prompt Templates...")
        prompts = await self._generate_prompts(sequence, algorithm_knowledge)
        metadata["prompts"] = prompts.model_dump()
        print(f"✅ Generated {len(prompts.prompts)} prompt templates")

        # Step 3: Generate Schemas
        print("\n📐 STEP 3: Designing Output Schemas...")
        schemas = await self._generate_schemas(sequence, algorithm_knowledge)
        metadata["schemas"] = schemas.model_dump()
        print(f"✅ Generated {len(schemas.schemas)} schemas")

        # Step 4: Validate & Align
        print("\n🔍 STEP 4: Validating & Aligning Plan...")

        # Determine final key (last stage's first write, or explicit from task)
        final_key = self._determine_final_key(sequence, task_description)

        plan = await self._validate_and_align(
            sequence, prompts, schemas, final_key, max_retries
        )
        metadata["final_key"] = final_key
        print(f"✅ Plan validated and aligned")
        print(f"   Final output key: '{final_key}'")

        print("\n✅ MULTI-AGENT PLANNING COMPLETE")
        print("=" * 60)

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
                print(f"✅ Validation passed on attempt {attempt + 1}")
                return plan

            print(f"⚠️  Validation issues found on attempt {attempt + 1}:")
            for error in validation_errors:
                print(f"   - {error}")

            if attempt < max_retries - 1:
                print(f"🔄 Retrying validation...")
                # Could add error feedback to next validation attempt
            else:
                print(f"⚠️  Max retries reached, returning plan with warnings")
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

        return errors


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
