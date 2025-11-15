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

# QUALITY REQUIREMENTS
- **Algorithmic Fidelity**: Stages must match canonical algorithm phases
- **Mathematical Precision**: Use exact terminology from algorithm knowledge
- **Implementation Ready**: Each stage must be executable with clear instructions
- **Validation Enabled**: Include criteria for verifying stage outputs
- **Context Flow**: Perfect reads/writes connectivity

# CRITICAL SUCCESS FACTORS
1. **Reference Algorithm Knowledge**: Use content from <CANONICAL_STAGES> to design stages
2. **Preserve Mathematical Objects**: Respect <KEY_MATHEMATICAL_OBJECTS> in schemas
3. **Detailed Prompts**: Create comprehensive, algorithm-specific prompt templates
4. **Perfect Flow**: Ensure each stage's reads are available from prior stages
5. **Validation Focus**: Include verification criteria in each prompt

The plan must be implementation-ready and algorithmically correct.
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