"""
Prompt Enhancement Agent

This module provides a single-responsibility agent that enhances stage prompts
with best practices to improve LLM execution quality while remaining algorithm-agnostic.

Design Principles:
- Generic: Works for any algorithm (PC, Dijkstra, sorting, etc.)
- Incremental: Enhances existing plans without changing architecture
- Focused: Only improves prompts, doesn't change stage structure
"""

from pydantic_ai import Agent
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from models import Plan, Stage


class PromptEnhancer:
    """
    Single-responsibility agent that enhances stage prompts with best practices.

    Key enhancements:
    1. Adds explicit conditional logic ("if X then Y, otherwise Z")
    2. Includes concrete examples and common mistakes
    3. Clarifies when no action is expected (e.g., "0 results is normal")
    4. Improves decision criteria with if-then tables
    5. Maintains algorithm-agnostic approach
    """

    def __init__(self):
        self.enhancer_agent = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are a prompt engineering specialist focused on improving LLM execution quality through clear, explicit instructions.

# TASK
Enhance a generic stage prompt by adding clarity, examples, and best practices while maintaining algorithm-agnostic structure.

# INPUT
You will receive:
- Stage ID: descriptive name of the stage
- Original prompt: basic task description
- Reads: input keys the stage uses
- Writes: output keys the stage produces
- Stage purpose: what this stage accomplishes

# ENHANCEMENT PRINCIPLES

## 1. Explicit Conditionals
Transform implicit instructions into explicit if-then logic.

**BEFORE**: "Orient edges using the rules"
**AFTER**: "Apply orientation rules IF specific graph patterns exist. If NO patterns match, keep edges unchanged."

## 2. Set Correct Expectations
Tell the LLM what "normal" outcomes look like.

Add statements like:
- "It is NORMAL and EXPECTED to find 0 results"
- "Many items may remain unchanged - this is CORRECT"
- "If conditions don't match, return input unchanged"

## 3. Common Mistakes Section
Add a section that explicitly states what NOT to do.

```
# COMMON MISTAKES TO AVOID
❌ Do NOT [common error]
❌ Do NOT [another error]
✅ DO [correct approach]
✅ DO [another correct approach]
```

## 4. Decision Criteria Tables
For decision-making stages, add clear if-then logic.

```
# DECISION LOGIC
IF condition A:
    return X
ELIF condition B:
    return Y
ELSE:
    return Z
```

## 5. Concrete Examples
Add mini-examples showing:
- Positive case (when conditions match)
- Negative case (when conditions don't match)
- Edge case handling

## 6. Preserve Algorithm-Agnostic Nature
**CRITICAL**: Do NOT add algorithm-specific logic (like "PC algorithm" or "Dijkstra").
Instead, use generic patterns:
- ✅ "If graph has directed edges, propagate; otherwise return unchanged"
- ❌ "Apply PC algorithm rules to orient edges"

# OUTPUT STRUCTURE

Return the enhanced prompt following this template:

```
# TASK
[Clear one-sentence description with explicit conditionals]

# INPUT DATA
{placeholder1}
{placeholder2}
[Keep original placeholders]

# CRITICAL UNDERSTANDING
[What the LLM needs to know about this stage's semantics]
[When is it OK to return empty/unchanged results]

# STEP-BY-STEP
1. [Clear, actionable step with conditional logic]
2. [Another step]
3. [Final step]

# DECISION CRITERIA (if applicable)
[Clear if-then logic for decision-making stages]

# EXAMPLES (if helpful)
[Mini worked example showing the process]

# OUTPUT
Return JSON with key(s): [list writes keys]

# COMMON MISTAKES TO AVOID
❌ [Specific mistake to avoid]
✅ [Correct approach]
```

# ENHANCEMENT GUIDELINES

## For Graph/Structure Modification Stages:
- Emphasize conditional application ("IF pattern exists THEN modify")
- State explicitly when no modifications are expected
- Warn against arbitrary changes

## For Identification/Detection Stages:
- State that finding 0 items is valid and normal
- Provide clear criteria for what counts as a match
- Include both positive and negative examples

## For Decision/Evaluation Stages:
- Provide explicit decision logic as if-then tables
- Clarify edge cases (what if multiple conditions apply?)
- Define key terms (e.g., "direct causation" vs "indirect")

## For Transformation Stages:
- Be explicit about what should and shouldn't change
- Provide format examples for complex transformations
- Warn against losing information during transformation

# QUALITY CRITERIA
Your enhanced prompt should:
1. Be 2-3x longer than the original (more detail = better execution)
2. Include at least one concrete example or decision table
3. Have a "Common Mistakes" section
4. Set correct expectations (when 0 results is OK)
5. Remain algorithm-agnostic (works for any algorithm)

# OUTPUT
Return ONLY the enhanced prompt text as a string, ready to replace the original.
"""
        )

    async def enhance_stage_prompt(self, stage: Stage, algorithm_context: str = "") -> str:
        """
        Enhance a single stage's prompt with best practices.

        Args:
            stage: The stage whose prompt needs enhancement
            algorithm_context: Optional context about the algorithm (generic description)

        Returns:
            Enhanced prompt string
        """

        enhancement_request = f"""
Enhance this stage prompt to improve LLM execution quality.

# STAGE INFORMATION
Stage ID: {stage.id}
Reads: {stage.reads}
Writes: {stage.writes}

# ORIGINAL PROMPT
{stage.prompt_template}

# ALGORITHM CONTEXT (Generic)
{algorithm_context if algorithm_context else "Generic multi-stage algorithmic workflow"}

# YOUR TASK
Enhance this prompt following the principles in your system prompt. Focus on:
1. Making conditionals explicit
2. Setting correct expectations
3. Adding decision criteria or examples
4. Including common mistakes section
5. Keeping it algorithm-agnostic

Return ONLY the enhanced prompt text.
"""

        result = await self.enhancer_agent.run(enhancement_request)
        return result.output

    async def enhance_plan(self, plan: Plan, algorithm_context: str = "") -> Plan:
        """
        Enhance all stage prompts in a plan.

        Args:
            plan: The plan to enhance
            algorithm_context: Optional generic context about the algorithm

        Returns:
            Plan with enhanced prompts
        """
        print("\n✨ ENHANCING STAGE PROMPTS...")
        print("=" * 60)

        enhanced_stages = []

        for i, stage in enumerate(plan.stages, 1):
            print(f"\n📝 Enhancing stage {i}/{len(plan.stages)}: {stage.id}")

            # Enhance the prompt
            enhanced_prompt = await self.enhance_stage_prompt(stage, algorithm_context)

            # Create new stage with enhanced prompt
            enhanced_stage = Stage(
                id=stage.id,
                reads=stage.reads,
                writes=stage.writes,
                prompt_template=enhanced_prompt,
                output_schema=stage.output_schema
            )

            enhanced_stages.append(enhanced_stage)

            print(f"   ✅ Enhanced ({len(stage.prompt_template)} → {len(enhanced_prompt)} chars)")

        # Create new plan with enhanced stages
        enhanced_plan = Plan(
            stages=enhanced_stages,
            final_key=plan.final_key
        )

        print("\n✅ All prompts enhanced")
        print(f"   Average prompt length: {sum(len(s.prompt_template) for s in enhanced_stages) // len(enhanced_stages)} chars")
        print("=" * 60)

        return enhanced_plan


# Example usage / testing
async def test_prompt_enhancement():
    """Test the prompt enhancer with a sample stage"""

    from models import Stage

    # Create a simple stage with generic prompt
    sample_stage = Stage(
        id="edge_orientation",
        reads=["graph", "input"],
        writes=["graph"],
        prompt_template="""# TASK
Apply propagation rules to orient edges in the graph.

# INPUT DATA
{graph}
{input}

# STEP-BY-STEP
1. Check for directed edges
2. Apply orientation rules
3. Update graph with oriented edges

# OUTPUT
Return JSON with key "graph" containing the updated graph.
""",
        output_schema={
            "type": "object",
            "properties": {
                "graph": {"type": "object"}
            },
            "required": ["graph"]
        }
    )

    enhancer = PromptEnhancer()

    print("🧪 Testing Prompt Enhancement")
    print("=" * 60)
    print("\n📝 ORIGINAL PROMPT:")
    print(sample_stage.prompt_template)
    print("\n" + "=" * 60)

    enhanced_prompt = await enhancer.enhance_stage_prompt(
        sample_stage,
        algorithm_context="Generic graph algorithm that processes nodes and edges"
    )

    print("\n✨ ENHANCED PROMPT:")
    print(enhanced_prompt)
    print("\n" + "=" * 60)

    print(f"\nLength increase: {len(sample_stage.prompt_template)} → {len(enhanced_prompt)} chars")
    print(f"Improvement: {(len(enhanced_prompt) / len(sample_stage.prompt_template) - 1) * 100:.1f}% longer")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_prompt_enhancement())
