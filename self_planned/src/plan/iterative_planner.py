from pydantic_ai import Agent
from typing import Dict, Any, List, Tuple
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from models import Plan


class IterativePlanner:
    """Enhanced planner with iterative refinement and self-reflection"""

    def __init__(self):
        # Plan generator with enhanced algorithmic awareness
        self.plan_generator = Agent(
            "openai:o3-mini",
            output_type=Plan,
            system_prompt="""
# ROLE
You are an expert algorithmic planning specialist who creates precise, mathematically-rigorous execution plans.

# TASK
Generate detailed execution plans that mirror canonical algorithm implementations.

# INPUT STRUCTURE
You will receive:
- **Algorithm knowledge** (with <CANONICAL_STAGES>, <KEY_MATHEMATICAL_OBJECTS>, etc.)
- **Task description** (specific problem to solve)

# PLANNING PRINCIPLES

## Mathematical Rigor
- Each stage must correspond to a specific algorithmic phase
- Preserve mathematical terminology and precision from the knowledge
- Include exact input/output specifications with mathematical objects
- Ensure algorithmic correctness and completeness

## Stage Design Quality
- **Reads/Writes Flow**: Perfect connectivity between stages
- **Granularity**: One mathematical transformation per stage
- **Validation**: Each stage must be verifiable
- **Specificity**: Detailed prompts with algorithm-specific instructions

## Output Format
Follow the Plan schema exactly:
```json
{
  "stages": [
    {
      "id": "descriptive_stage_name",
      "reads": ["input_key1", "input_key2"],
      "writes": ["output_key1", "output_key2"],
      "prompt_template": "Detailed mathematical instructions using {input_key1} and {input_key2}...",
      "output_schema": {detailed JSON schema matching writes keys}
    }
  ],
  "final_key": "key_containing_final_result"
}
```

# PROMPT TEMPLATE CONSTRUCTION
For each stage's `prompt_template`, use this INPUT-FOCUSED structure:

```
# TASK
[One clear sentence describing what to do with the input data]

# INPUT DATA
{input_data_keys}

# STEP-BY-STEP
1. [Simple, input-focused step]
2. [Simple, input-focused step]
3. [Simple, input-focused step]

# OUTPUT
Return JSON with the specified keys.

# CRITICAL RULES
- Work ONLY with the provided input data
- Use ONLY the variable names that appear in the input
- DO NOT use generic variables (X, Y, Z) or textbook examples
- DO NOT add variables not mentioned in the input
- DO NOT remove variables that exist in the input
```

# QUALITY REQUIREMENTS
- **Input-Focused Prompts**: Use the input-focused template structure above
- **Data-Driven Tasks**: Each stage works only with provided input data
- **Variable Preservation**: Ensure all input variables flow through stages
- **Schema Compatibility**: Output schemas must match next stage inputs

# CRITICAL SUCCESS FACTORS
1. **Base on Algorithm Knowledge**: Use <CANONICAL_STAGES> to design stage sequence
2. **Input-Only Execution**: Stage prompts reference ONLY input data, no algorithmic theory
3. **Preserve All Data**: Every variable from input must appear in every relevant stage
4. **Simple Instructions**: Make each stage's task clear and data-focused

# EXECUTION PHILOSOPHY
- **Planning Phase**: Use algorithm knowledge to design correct stage sequence
- **Execution Phase**: Stage prompts focus purely on transforming input data
- **No Algorithm Theory in Prompts**: Remove mathematical definitions from stage templates

The plan must preserve all input variables and avoid algorithmic theory in execution prompts.
"""
        )

        # Plan critic for self-reflection
        self.plan_critic = Agent(
            "openai:gpt-4o-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are a critical plan evaluator specializing in algorithmic correctness and implementation quality.

# TASK
Analyze execution plans for mathematical rigor, algorithmic correctness, and implementation quality.

# EVALUATION CRITERIA

## Algorithmic Correctness (0-10)
- Do stages match canonical algorithm phases?
- Is the mathematical flow logically correct?
- Are algorithm-specific requirements preserved?

## Implementation Quality (0-10)
- Are prompt templates detailed and specific?
- Do schemas properly define mathematical objects?
- Are reads/writes properly connected?

## Prompt Template Quality (0-10)
- Does each prompt template follow the required structured format?
- Are role definitions clear and domain-specific?
- Do prompts include step-by-step processes with validation?
- Are mathematical requirements and critical success factors specified?
- Do prompts provide clear input/output specifications?

## Mathematical Precision (0-10)
- Is algorithmic terminology used correctly?
- Are mathematical definitions preserved?
- Do stages respect mathematical constraints?

## Completeness (0-10)
- Are all algorithmic phases included?
- Is the plan missing any critical steps?
- Does it cover edge cases and validation?

# OUTPUT FORMAT
Provide detailed analysis in this structure:

## QUALITY SCORES
- Algorithmic Correctness: X/10
- Implementation Quality: X/10
- Prompt Template Quality: X/10
- Mathematical Precision: X/10
- Completeness: X/10
- **OVERALL SCORE: X/10**

## STRENGTHS
- [Specific strengths identified]

## CRITICAL ISSUES
- [Major problems that need fixing]

## IMPROVEMENT SUGGESTIONS
- [Specific, actionable improvements]

## ALGORITHMIC FIDELITY CHECK
- [How well does the plan match the canonical algorithm?]

Be thorough, critical, and provide actionable feedback for improvement.
"""
        )

        # Plan refiner for iterative improvement
        self.plan_refiner = Agent(
            "openai:o3-mini",
            output_type=Plan,
            system_prompt="""
# ROLE
You are a plan refinement expert who improves execution plans based on critical feedback.

# TASK
Take an existing plan and critical analysis, then generate an improved version addressing all identified issues.

# INPUT
- **Original Plan**: The plan to improve
- **Critical Analysis**: Detailed feedback with specific issues and suggestions
- **Algorithm Knowledge**: Reference material for algorithmic correctness

# IMPROVEMENT PRINCIPLES
1. **Address All Issues**: Fix every problem identified in the critical analysis
2. **Enhance Quality**: Improve mathematical precision and implementation detail
3. **Preserve Strengths**: Keep what's working well from the original plan
4. **Algorithmic Fidelity**: Ensure closer alignment with canonical algorithm phases
5. **Implementation Focus**: Make it more executable with better prompts and schemas

# REFINEMENT TARGETS
- **Fix Connectivity Issues**: Resolve reads/writes flow problems
- **Enhance Prompt Quality**: Add mathematical rigor and specific instructions
- **Improve Schemas**: Better define mathematical objects and structures
- **Add Validation**: Include verification criteria in prompts
- **Algorithm Alignment**: Better match canonical algorithmic phases

# OUTPUT
Generate an improved Plan that addresses the critical feedback while maintaining algorithmic correctness.

Focus on making the plan more executable, mathematically precise, and algorithmically faithful.
"""
        )

        # Schema refiner for mandatory schema improvement
        self.schema_refiner = Agent(
            "openai:o3-mini",
            output_type=Plan,
            system_prompt="""
# ROLE
You are a schema compatibility specialist focused on ensuring data consistency across execution stages.

# TASK
Refine the schemas of a plan to ensure perfect compatibility between stages while preserving algorithmic correctness.

# CRITICAL OBJECTIVES

## Schema Consistency
- **Format Compatibility**: Ensure stage outputs match next stage input expectations
- **Entity Preservation**: Maintain all critical entities (variables, nodes, etc.) across pipeline
- **Structure Consistency**: Use compatible data formats between connected stages

## Entity Preservation Requirements
- **Variables**: If algorithm works with variables (A, B, C, D, E), preserve them throughout
- **Graph Structures**: Maintain consistent node/edge representations
- **Mathematical Objects**: Preserve all algorithmic objects without loss

## Data Format Standardization
- **Graph Representations**: Use consistent format (adjacency list vs structured) across stages
- **Entity References**: Ensure entities are referenced consistently in all schemas
- **Schema Clarity**: Make schemas explicit about data structure and content

# REFINEMENT PROCESS

1. **Analyze Entity Flow**: Identify what entities/objects flow between stages
2. **Standardize Formats**: Ensure compatible data formats across stage boundaries
3. **Preserve Critical Data**: Guarantee no loss of algorithmic information
4. **Validate Compatibility**: Ensure each stage output satisfies next stage input requirements

# OUTPUT REQUIREMENTS
- **Same Plan Structure**: Preserve all stages, reads/writes, and prompt templates
- **Improved Schemas**: Only modify output_schema fields for compatibility
- **Entity Preservation**: Ensure critical entities are maintained across all relevant stages
- **Format Consistency**: Use consistent data formats between connected stages

# CRITICAL SUCCESS FACTORS
- All entities present in input must flow through to final stages (no entity loss)
- Compatible data formats between all stage transitions
- Schemas must be clear and executable by LLM agents
- Preserve the algorithmic logic while improving technical compatibility

Focus on making schemas technically compatible while maintaining perfect algorithmic fidelity.
"""
        )

    async def generate_iterative_plan(self,
                                    task_description: str,
                                    algorithm_knowledge: str,
                                    max_iterations: int = 3,
                                    target_score: float = 8.0) -> Tuple[Plan, List[Dict[str, Any]]]:
        """Generate a high-quality plan through iterative refinement"""

        print(f"🔄 Starting iterative planning (max {max_iterations} iterations, target score: {target_score})")

        iteration_history = []
        current_plan = None

        for iteration in range(1, max_iterations + 1):
            print(f"\n📝 ITERATION {iteration}")

            if iteration == 1:
                # Generate initial plan
                print("🎯 Generating initial plan...")
                current_plan = await self._generate_initial_plan(task_description, algorithm_knowledge)
            else:
                # Refine based on previous feedback
                print("🔧 Refining plan based on feedback...")
                previous_feedback = iteration_history[-1]["feedback"]
                current_plan = await self._refine_plan(current_plan, previous_feedback, algorithm_knowledge) # pyright: ignore[reportArgumentType]

            # Always refine schemas for compatibility
            print("🔧 Refining schemas for compatibility...")
            current_plan = await self._refine_schemas(current_plan, algorithm_knowledge)

            # Evaluate the plan
            print("📊 Evaluating plan quality...")
            feedback = await self._evaluate_plan(current_plan, algorithm_knowledge, task_description)

            iteration_info = {
                "iteration": iteration,
                "plan": current_plan.model_dump() if current_plan else None,
                "feedback": feedback,
                "num_stages": len(current_plan.stages) if current_plan else 0
            }
            iteration_history.append(iteration_info)

            # Extract overall score
            overall_score = self._extract_score_from_feedback(feedback)
            print(f"📈 Overall Score: {overall_score}/10")

            # Check if we've reached target quality
            if overall_score >= target_score:
                print(f"✅ Target quality reached! ({overall_score}/10 >= {target_score}/10)")
                break
            elif iteration < max_iterations:
                print(f"🔄 Score below target, continuing to iteration {iteration + 1}...")
            else:
                print(f"🏁 Max iterations reached. Final score: {overall_score}/10")

        print(f"\n🎯 Planning completed after {len(iteration_history)} iterations")
        return current_plan, iteration_history # pyright: ignore[reportReturnType]

    async def _generate_initial_plan(self, task_description: str, algorithm_knowledge: str) -> Plan:
        """Generate the initial plan"""

        planning_prompt = f"""
# PLANNING REQUEST
Create an execution plan for the following task using the provided algorithm knowledge.

## TASK DESCRIPTION
{task_description}

## ALGORITHM KNOWLEDGE
{algorithm_knowledge}

# YOUR TASK
Generate a detailed execution plan that implements the canonical algorithm stages described in the knowledge.

**CRITICAL**: Use the content from <CANONICAL_STAGES> to design your stages. Each stage should correspond to a specific algorithmic phase.

Focus on mathematical rigor, implementation clarity, and algorithmic correctness.
"""

        result = await self.plan_generator.run(planning_prompt)
        return result.output

    async def _evaluate_plan(self, plan: Plan, algorithm_knowledge: str, task_description: str) -> str:
        """Evaluate plan quality with detailed feedback"""

        evaluation_prompt = f"""
# PLAN EVALUATION REQUEST

## TASK BEING SOLVED
{task_description}

## ALGORITHM KNOWLEDGE REFERENCE
{algorithm_knowledge}

## PLAN TO EVALUATE
{json.dumps(plan.model_dump(), indent=2)}

# YOUR TASK
Provide a thorough critical analysis of this execution plan. Focus on algorithmic correctness, implementation quality, and mathematical precision.

Compare the plan stages against the <CANONICAL_STAGES> from the algorithm knowledge to assess algorithmic fidelity.
"""

        result = await self.plan_critic.run(evaluation_prompt)
        return result.output

    async def _refine_plan(self, current_plan: Plan, feedback: str, algorithm_knowledge: str) -> Plan:
        """Refine the plan based on critical feedback"""

        refinement_prompt = f"""
# PLAN REFINEMENT REQUEST

## ORIGINAL PLAN
{json.dumps(current_plan.model_dump(), indent=2)}

## CRITICAL ANALYSIS & FEEDBACK
{feedback}

## ALGORITHM KNOWLEDGE REFERENCE
{algorithm_knowledge}

# YOUR TASK
Generate an improved version of the plan that addresses all issues identified in the critical analysis.

Focus on:
1. Fixing all problems mentioned in the feedback
2. Improving mathematical precision and algorithmic fidelity
3. Enhancing implementation quality and detail
4. Maintaining what's working well

Reference the <CANONICAL_STAGES> in the algorithm knowledge to ensure algorithmic correctness.
"""

        result = await self.plan_refiner.run(refinement_prompt)
        return result.output

    async def _refine_schemas(self, plan: Plan, algorithm_knowledge: str) -> Plan:
        """Refine schemas for compatibility (always runs)"""

        refinement_prompt = f"""
# SCHEMA REFINEMENT REQUEST

## CURRENT PLAN
{json.dumps(plan.model_dump(), indent=2)}

## ALGORITHM KNOWLEDGE REFERENCE
{algorithm_knowledge}

# YOUR TASK
Refine the schemas in this plan to ensure perfect compatibility between stages while preserving algorithmic correctness.

## FOCUS AREAS
1. **Entity Preservation**: Ensure all variables/entities from the algorithm flow through all relevant stages
2. **Format Consistency**: Use compatible data formats between connected stages
3. **Schema Clarity**: Make schemas explicit and executable

## CRITICAL REQUIREMENTS
- Preserve all algorithmic entities (variables, nodes, etc.) throughout the pipeline
- Use consistent data format patterns between stages
- Ensure each stage output schema satisfies the next stage's input requirements
- Only modify output_schema fields - preserve all other plan elements

Generate a refined version of this plan with improved schema compatibility.
"""

        result = await self.schema_refiner.run(refinement_prompt)
        return result.output

    async def generate_simple_plan(self, task_description: str, algorithm_knowledge: str) -> Plan:
        """
        Generate a plan using a simple, single-request approach.

        This is a backup method that creates a plan without iterative refinement,
        designed for performance testing and scenarios where the iterative approach
        may be too resource-intensive.

        Args:
            task_description: Description of the task to be planned
            algorithm_knowledge: Canonical algorithm knowledge to base the plan on

        Returns:
            Simple plan generated in a single request
        """
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

CRITICAL Prompt Template Structure:
EVERY prompt_template MUST follow this exact pattern:

```
# TASK
[Clear description of algorithmic step referencing the input data below]

# INPUT DATA
{{placeholder_for_each_read_key}}  (list each read key exactly once)

# STEP-BY-STEP
1. [Specific instruction that refers to "the input data provided above"]
2. [Specific instruction that refers to "the variables in the input data"]
3. [Specific instruction that refers to "the relationships described in the input"]

# OUTPUT
Return JSON with the specified keys.
```

Contract:
- Each stage defines:
  - reads[]: context keys it expects (subset of keys available so far),
  - writes[]: new/updated context keys it will produce (non-empty),
  - prompt_template: detailed template that MUST include {{placeholder}} for EVERY key in reads[],
  - output_schema: strict JSON Schema describing exactly the keys you write,
- Context model:
  - The initial context contains ONLY the keys named by the caller.
  - After each stage, ONLY keys in writes[] may be added/updated in context.
- Reads/writes discipline:
  - reads[] must be a subset of available keys (initial keys ∪ prior writes).
  - Every writes[] key MUST appear in the stage's output JSON (per output_schema).
- The plan MUST set final_key to the context key that represents the final answer/result.
- All stage outputs must be STRICT JSON (no extra text).

CRITICAL SCHEMA RULES:
- If writes: ["key1"] → output_schema must have properties: {{"key1": {{...}}}}
- If writes: ["key1", "key2"] → output_schema must have properties: {{"key1": {{...}}, "key2": {{...}}}}
- The output_schema.properties keys MUST exactly match the writes array
- NO EXCEPTIONS: writes and output_schema.properties must be perfectly aligned

SCHEMA GENERATION GUIDELINES:
- Use the DATA FORMATS section from the algorithm knowledge to generate detailed schemas
- Replace generic {{"type": "object"}} with specific structures when DATA FORMATS provides them
- For mathematical objects (graphs, sepsets, etc.), use the detailed structure from DATA FORMATS
- If DATA FORMATS doesn't specify a structure, fall back to descriptive object schema

CORRECT EXAMPLES:
✅ writes: ["skeleton"] → output_schema: {{"type": "object", "properties": {{"skeleton": {{"type": "object"}}}}, "required": ["skeleton"]}}
✅ writes: ["graph", "sepsets"] → output_schema: {{"type": "object", "properties": {{"graph": {{"type": "object"}}, "sepsets": {{"type": "object"}}}}, "required": ["graph", "sepsets"]}}
✅ WITH DATA FORMATS: writes: ["graph"] → output_schema: {{"type": "object", "properties": {{"graph": {{"type": "object", "properties": {{"nodes": {{"type": "array", "items": {{"type": "string"}}}}, "edges": {{"type": "array", "items": {{"type": "array", "items": {{"type": "string"}}, "minItems": 2, "maxItems": 2}}}}}}, "required": ["nodes", "edges"]}}}}, "required": ["graph"]}}

INCORRECT EXAMPLES:
❌ writes: ["skeleton"] + output_schema.properties: {{"graph": ..., "sepsets": ...}} // WRONG - mismatch
❌ writes: ["result"] + output_schema.properties: {{"final_decision": ...}} // WRONG - different keys

PROMPT TEMPLATE ALIGNMENT:
- If writes: ["skeleton"] → prompt must say "Return JSON: {{\\"skeleton\\": {{...}}}}"
- If writes: ["graph", "sepsets"] → prompt must say "Return JSON: {{\\"graph\\": ..., \\"sepsets\\": ...}}"
"""

        simple_planner = Agent("openai:o3-mini", output_type=Plan, system_prompt=system_prompt)

        planning_prompt = f"Task: {task_description}"
        result = await simple_planner.run(planning_prompt)

        # Validate the generated plan
        plan = result.output
        self._validate_plan_templates(plan)

        return plan

    def _validate_plan_templates(self, plan: Plan) -> None:
        """
        Validate that all prompt templates properly use placeholders for their reads keys
        and that output schemas match writes declarations.
        Raises ValueError if validation fails.
        """
        import re

        for stage in plan.stages:
            # 1. Validate placeholder usage
            placeholders = set(re.findall(r'\{(\w+)\}', stage.prompt_template))
            missing_placeholders = set(stage.reads) - placeholders
            if missing_placeholders:
                raise ValueError(
                    f"Stage '{stage.id}' is missing placeholders for reads keys: {missing_placeholders}. "
                    f"Reads: {stage.reads}, Found placeholders: {placeholders}. "
                    f"Each reads key must have a corresponding {{key}} placeholder in the prompt_template."
                )

            # 2. Validate schema-writes consistency
            schema_keys = set(stage.output_schema.get('properties', {}).keys())
            writes_keys = set(stage.writes)

            if schema_keys != writes_keys:
                raise ValueError(
                    f"Stage '{stage.id}' has mismatched writes and output schema. "
                    f"Writes: {stage.writes}, Schema properties: {list(schema_keys)}. "
                    f"The output_schema.properties keys must exactly match the writes array."
                )

            print(f"✅ Validation passed for stage '{stage.id}': placeholders={placeholders}, schema_keys={schema_keys}")

    def _extract_score_from_feedback(self, feedback: str) -> float:
        """Extract the overall score from feedback text"""
        try:
            # Look for patterns like "OVERALL SCORE: 7.5/10" or "OVERALL SCORE: 7.5"
            import re
            patterns = [
                r"OVERALL SCORE:\s*(\d+(?:\.\d+)?)/10",
                r"OVERALL SCORE:\s*(\d+(?:\.\d+)?)",
                r"Overall Score:\s*(\d+(?:\.\d+)?)/10",
                r"Overall Score:\s*(\d+(?:\.\d+)?)"
            ]

            for pattern in patterns:
                match = re.search(pattern, feedback, re.IGNORECASE)
                if match:
                    return float(match.group(1))

            # Fallback: look for any score pattern
            score_pattern = r"(\d+(?:\.\d+)?)/10"
            matches = re.findall(score_pattern, feedback)
            if matches:
                # Take the last score found (likely overall)
                return float(matches[-1])

            return 5.0  # Default middle score if no pattern found
        except:  # noqa: E722
            return 5.0  # Safe fallback


# Test function
async def test_iterative_planning():
    """Test the iterative planning system"""

    planner = IterativePlanner()

    task_description = """
Task: Given a natural-language input containing a Premise and a Hypothesis, decide whether the Hypothesis is True or False under the Peter-Clark (PC) algorithm.

Input available in context: 'input' (contains premise with variables, correlations, conditional independencies, and hypothesis).

CRITICAL OUTPUT FORMAT: The final stage must output ONLY a boolean value (true or false).
"""

    # Mock algorithm knowledge (in real usage, this comes from enhanced extractor)
    algorithm_knowledge = """
## <CANONICAL_STAGES>
1. **Skeleton Discovery**: Build undirected graph from correlations, remove edges based on independence
2. **V-Structure Identification**: Find colliders using separation sets
3. **Edge Orientation**: Apply Meek's rules to orient remaining edges
4. **Hypothesis Evaluation**: Test hypothesis against final CPDAG

## <KEY_MATHEMATICAL_OBJECTS>
- Undirected Graph: nodes and edges
- Separation Sets: conditional independence information
- CPDAG: completed partially directed acyclic graph
"""

    print("🧪 Testing Iterative Planning System")
    print("=" * 50)

    try:
        final_plan, history = await planner.generate_iterative_plan(
            task_description=task_description,
            algorithm_knowledge=algorithm_knowledge,
            max_iterations=2,
            target_score=7.0
        )

        print("\n📋 FINAL PLAN SUMMARY:")
        print(f"Number of stages: {len(final_plan.stages)}")
        print(f"Final key: {final_plan.final_key}")
        print(f"Iterations used: {len(history)}")

        print("\n🎯 ITERATION HISTORY:")
        for i, iteration in enumerate(history, 1):
            score = planner._extract_score_from_feedback(iteration["feedback"])
            print(f"  Iteration {i}: {score}/10 ({iteration['num_stages']} stages)")

        return final_plan, history

    except Exception as e:
        print(f"❌ Error during iterative planning: {e}")
        return None, []


if __name__ == "__main__":
    asyncio.run(test_iterative_planning())
