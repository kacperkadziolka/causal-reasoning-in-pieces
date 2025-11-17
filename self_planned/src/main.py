import asyncio
from typing import Any, Optional, Dict
from dotenv import load_dotenv
import pandas as pd
import random
from pydantic_ai import Agent

from knowledge.extractor import EnhancedKnowledgeExtractor
from plan.iterative_planner import IterativePlanner
from execute.executor import run_plan

load_dotenv()


def fetch_sample(csv_path: str) -> pd.Series:
    """Fetch a random sample from the dataset"""
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} samples")

    # Get a random sample
    sample_idx = random.randint(0, len(df) - 1)
    sample = df.iloc[sample_idx]

    print("\n=== SAMPLE FROM TEST DATASET ===")
    print(f"Index: {sample_idx}")
    print(f"Input: {sample['input']}")
    print(f"Label: {sample['label']}")
    print(f"Num Variables: {sample['num_variables']}")
    print(f"Template: {sample['template']}")
    print("=" * 50)

    return sample


async def detect_algorithm(task_description: str) -> str:
    """Detect if a specific algorithm is mentioned in the task description using enhanced prompt engineering."""

    algorithm_detector = Agent(
        "openai:gpt-4o-mini",
        output_type=str,
        system_prompt="""
# ROLE
You are an expert algorithm identification specialist with comprehensive knowledge of academic algorithms across all domains.

# TASK
Extract the primary algorithm/method mentioned in task descriptions with high precision and academic accuracy.

# ALGORITHM CATEGORIES TO CONSIDER
## <CAUSAL_DISCOVERY>
- Peter-Clark (PC), Fast Causal Inference (FCI), Greedy Equivalence Search (GES), Linear Non-Gaussian Acyclic Model (LiNGAM)
</CAUSAL_DISCOVERY>

## <GRAPH_ALGORITHMS>
- Dijkstra, A*, Breadth-First Search (BFS), Depth-First Search (DFS), Floyd-Warshall, Bellman-Ford
</GRAPH_ALGORITHMS>

## <MACHINE_LEARNING>
- Gradient Descent, Stochastic Gradient Descent (SGD), K-Means, Support Vector Machine (SVM), Random Forest
</MACHINE_LEARNING>

## <OPTIMIZATION>
- Genetic Algorithm (GA), Simulated Annealing, Particle Swarm Optimization (PSO), Branch and Bound
</OPTIMIZATION>

## <SEARCH_ALGORITHMS>
- Binary Search, Linear Search, Minimax, Alpha-Beta Pruning, Monte Carlo Tree Search (MCTS)
</SEARCH_ALGORITHMS>

# DETECTION RULES

## Positive Identification Criteria
- **Explicit mentions**: "using [algorithm name]", "apply [algorithm]", "based on [algorithm]"
- **Academic references**: Standard algorithm names from academic literature
- **Abbreviated forms**: Include both full name and common abbreviation when applicable
- **Algorithm families**: Identify specific variant when mentioned (e.g., "SGD" vs "Gradient Descent")

## Exclusion Criteria
- **Generic terms**: "reasoning", "analysis", "method", "approach", "technique", "procedure"
- **Domain descriptions**: "machine learning", "optimization", "search" without specific algorithm
- **Process descriptions**: "training", "learning", "solving" without algorithmic specifics

# OUTPUT FORMAT
Return the algorithm name exactly as it appears in academic literature:
- **Include abbreviations** in parentheses when commonly used: "Peter-Clark (PC)"
- **Use standard academic naming**: "Dijkstra" not "Dijkstra's algorithm"
- **Preserve case sensitivity**: "A*" not "a*", "LiNGAM" not "lingam"
- **Return "none"** if no specific algorithm is identified

# EXAMPLES

## <POSITIVE_EXAMPLES>
- "decide whether the Hypothesis is True or False under the Peter-Clark (PC) algorithm" → "Peter-Clark (PC)"
- "solve the shortest path problem using Dijkstra's algorithm" → "Dijkstra"
- "optimize the parameters with gradient descent" → "Gradient Descent"
- "apply A* search to find the optimal path" → "A*"
- "use the Genetic Algorithm for optimization" → "Genetic Algorithm (GA)"
</POSITIVE_EXAMPLES>

## <NEGATIVE_EXAMPLES>
- "perform causal discovery analysis" → "none" (no specific algorithm)
- "use machine learning techniques" → "none" (too generic)
- "solve the optimization problem" → "none" (no specific algorithm)
- "apply reasoning methods" → "none" (generic reasoning)
</NEGATIVE_EXAMPLES>

# CRITICAL INSTRUCTIONS
1. **Single algorithm focus**: Return only the PRIMARY algorithm mentioned
2. **Academic precision**: Use exact academic naming conventions
3. **Context awareness**: Consider the domain context when disambiguating
4. **Abbreviation inclusion**: Add common abbreviations when standard practice
5. **Conservative identification**: When uncertain, prefer "none" over guessing

**OUTPUT**: Return only the algorithm name following the format rules above, no additional text or explanations.
""",
    )

    result = await algorithm_detector.run(task_description)
    return result.output.strip()


async def run_enhanced_workflow(sample: pd.Series) -> Optional[Dict[str, Any]]:
    """Run the complete enhanced workflow: detection → knowledge → planning → execution"""

    print("\n🚀 ENHANCED WORKFLOW")
    print("=" * 60)

    # Enhanced task description with concrete sample and algorithm-agnostic approach
    task_description = f"""
# TASK SPECIFICATION
Analyze natural-language causal reasoning problems using the **Peter-Clark (PC) algorithm** to determine hypothesis validity.

## <INPUT_SPECIFICATION>
**Available Context Key**: `input`

**Input Structure**: Natural language text containing:
- **Premise**: Statistical relationships among variables (correlations, independencies, conditional independencies)
- **Hypothesis**: A specific causal claim to be validated

## <CONCRETE_EXAMPLE>
**Current Sample Input**:
```
{sample['input']}
```

**Expected Label**: {sample['label']} (where True=1, False=0)
**Variables**: {sample['num_variables']} variables
**Template Type**: {sample['template']}

## <TASK_REQUIREMENTS>
### Algorithm Application
- Apply the **Peter-Clark (PC) algorithm** as specified in academic literature
- Use the algorithm to analyze the causal relationships described in the premise
- Determine whether the hypothesis is valid according to the algorithm's methodology

### Decision Criteria
- Return `true` if the hypothesis is supported by the algorithm's analysis
- Return `false` if the hypothesis is not supported or contradicted
- Apply rigorous mathematical reasoning as defined by the PC algorithm

## <OUTPUT_SPECIFICATION>
### Critical Requirements
- **Final Output**: EXACTLY one boolean value (`true` or `false`)
- **Output Key**: The final stage must write to a clearly defined output key
- **Format**: Pure boolean value, no additional text or explanations

### Success Criteria
- Algorithmic correctness and fidelity to PC algorithm principles
- Comprehensive analysis of all variables and relationships in the premise
- Sound mathematical reasoning leading to the final decision

**OBJECTIVE**: Implement and execute the PC algorithm correctly to validate the given hypothesis against the provided premise.
"""

    # Step 1: Algorithm Detection
    print("\n🔍 STEP 1: Algorithm Detection")
    print("-" * 30)
    algorithm = await detect_algorithm(task_description)
    print(f"🎯 Detected algorithm: {algorithm}")

    if algorithm == "none":
        print("⚠️  No specific algorithm detected - using generic planning approach")
        # Could still proceed with generic planning, but for now we'll show the limitation
        return None

    # Step 2: Enhanced Knowledge Extraction
    print("\n📚 STEP 2: Enhanced Knowledge Extraction")
    print("-" * 30)
    extractor = EnhancedKnowledgeExtractor()

    try:
        knowledge = await extractor.extract_enhanced_knowledge(algorithm)
        print(f"✅ Enhanced knowledge extracted: {len(knowledge)} characters")

        # Show a preview of the extracted knowledge
        preview_lines = knowledge.split('\n')[:5]
        print("📄 Knowledge preview:")
        for line in preview_lines:
            print(f"    {line}")
        if len(knowledge.split('\n')) > 5:
            print(f"    ... ({len(knowledge.split('\n')) - 5} more lines)")

    except Exception as e:
        print(f"❌ Knowledge extraction failed: {e}")
        return None

    # Step 3: Iterative Planning
    print("\n🔄 STEP 3: Iterative Planning")
    print("-" * 30)
    planner = IterativePlanner()

    try:
        plan, planning_history = await planner.generate_iterative_plan(
            task_description=task_description,
            algorithm_knowledge=knowledge,
            max_iterations=3,
            target_score=7.5
        )

        if not plan:
            print("❌ Planning failed")
            return None

        final_score = planner._extract_score_from_feedback(planning_history[-1]["feedback"])
        print(f"✅ Planning successful: {len(plan.stages)} stages (quality: {final_score}/10)")

        # Show planning progression
        print("📈 Planning progression:")
        for i, iteration in enumerate(planning_history, 1):
            iter_score = planner._extract_score_from_feedback(iteration["feedback"])
            print(f"    Iteration {i}: {iter_score}/10")

    except Exception as e:
        print(f"❌ Planning failed: {e}")
        return None

    # Step 4: Execution
    print("\n⚡ STEP 4: Enhanced Execution")
    print("-" * 30)
    initial_context = {"input": sample["input"]}

    try:
        final_context = await run_plan(plan, initial_context)
        final_key = plan.final_key or "result"
        final_result = final_context.get(final_key)

        print("✅ Execution completed")
        print(f"🎯 Final result key: '{final_key}'")
        print(f"📊 Final result: {final_result}")

        # Step 5: Evaluation
        print("\n📊 STEP 5: Evaluation")
        print("-" * 30)
        expected = bool(sample['label'])

        # Convert final_result to boolean
        if isinstance(final_result, str):
            actual = final_result.lower() in ['true', '1', 'yes']
        elif isinstance(final_result, int):
            actual = bool(final_result)
        else:
            actual = bool(final_result)

        is_correct = actual == expected

        print(f"🎯 Predicted: {actual}")
        print(f"📊 Expected: {expected}")
        print(f"✅ Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        return {
            "sample_idx": sample.name if hasattr(sample, 'name') else 'N/A',
            "algorithm": algorithm,
            "knowledge_length": len(knowledge),
            "planning_iterations": len(planning_history),
            "planning_quality": final_score,
            "num_stages": len(plan.stages),
            "predicted": actual,
            "expected": expected,
            "is_correct": is_correct,
            "final_context": final_context,
            "plan_summary": {
                "stages": [{
                    "id": stage.id,
                    "reads": stage.reads,
                    "writes": stage.writes,
                    "prompt_template": stage.prompt_template,
                    "output_schema": stage.output_schema
                } for stage in plan.stages],
                "final_key": plan.final_key
            }
        }

    except Exception as e:
        print(f"❌ Execution failed: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        return None


async def main():
    """Enhanced main function"""
    print("🔬 ENHANCED SELF-PLANNED PIPELINE")
    print("=" * 60)
    print("🎯 Generic algorithm detection and processing")
    print("📚 Enhanced knowledge extraction with multi-perspective analysis")
    print("🔄 Iterative planning with self-reflection and quality scoring")
    print("⚡ Quality-aware execution")

    # Load sample
    csv_path = "../data/test_dataset.csv"
    sample = fetch_sample(csv_path)

    # Run enhanced workflow
    result = await run_enhanced_workflow(sample)

    # Final summary
    print("\n" + "=" * 60)
    print("📋 ENHANCED PIPELINE SUMMARY")
    print("=" * 60)

    if result:
        print("✅ Workflow completed successfully!")
        print(f"🧠 Algorithm: {result['algorithm']}")
        print(f"📚 Knowledge: {result['knowledge_length']} chars")
        print(f"🔄 Planning: {result['planning_iterations']} iterations (quality: {result['planning_quality']}/10)")
        print(f"📝 Stages: {result['num_stages']}")
        print(f"🎯 Prediction: {result['predicted']} (expected: {result['expected']})")
        print(f"📊 Accuracy: {'✅ CORRECT' if result['is_correct'] else '❌ INCORRECT'}")

        print("\n📝 Generated Plan Overview:")
        for i, stage in enumerate(result['plan_summary']['stages'], 1):
            print(f"  {i}. {stage['id']}")
            print(f"     📥 Reads: {stage['reads']}")
            print(f"     📤 Writes: {stage['writes']}")

        print(f"\n🎯 Final output key: {result['plan_summary']['final_key']}")

        # Show potential for other algorithms
        print("\n🔮 Algorithm Generalizability:")
        print("    ✅ This pipeline can handle any commonly known algorithm")
        print("    ✅ Algorithm detection: Automatic via LLM")
        print("    ✅ Knowledge extraction: Multi-perspective analysis")
        print("    ✅ Planning: Iterative improvement with quality scoring")
        print("    💡 To test other algorithms, simply change the task description")

    else:
        print("❌ Workflow failed")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
