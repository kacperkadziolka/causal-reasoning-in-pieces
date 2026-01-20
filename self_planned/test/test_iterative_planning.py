import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path BEFORE importing our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.plan.iterative_planner import IterativePlanner  # noqa: E402
from src.knowledge.extractor import EnhancedKnowledgeExtractor  # noqa: E402

load_dotenv()

async def test_full_enhanced_pipeline():
    """Test the full enhanced pipeline: knowledge extraction + iterative planning"""

    print("🔬 Testing Enhanced Pipeline: Knowledge + Iterative Planning")
    print("=" * 70)

    # Step 1: Extract enhanced knowledge
    print("\n📚 STEP 1: Enhanced Knowledge Extraction")
    print("-" * 40)

    extractor = EnhancedKnowledgeExtractor()
    algorithm = "Peter-Clark (PC)"

    print(f"🎯 Extracting knowledge for: {algorithm}")
    knowledge = await extractor.extract_enhanced_knowledge(algorithm)
    print(f"✅ Knowledge extracted: {len(knowledge)} characters")

    # Step 2: Generate iterative plan
    print("\n🔄 STEP 2: Iterative Planning")
    print("-" * 40)

    task_description = """
Task: Given a natural-language input containing a Premise and a Hypothesis, decide whether the Hypothesis is True or False under the Peter-Clark (PC) algorithm.

- PC is a constraint-based causal discovery method that infers a causal equivalence class (CPDAG) from observational (in)dependence information.
- Before deciding, reconstruct a global causal structure over all variables mentioned in the Premise.
- Return True only if the claim holds in every DAG in the Markov equivalence class implied by the Premise; otherwise return False.

Input available in context: 'input' (contains premise with variables, correlations, conditional independencies, and hypothesis).

CRITICAL OUTPUT FORMAT: The final stage must output ONLY a boolean value (true or false).
"""

    planner = IterativePlanner()

    print("🎯 Starting iterative planning...")
    final_plan, history = await planner.generate_iterative_plan(
        task_description=task_description,
        algorithm_knowledge=knowledge,
        max_iterations=2,  # Keep it short for testing
        target_score=7.0
    )

    # Step 3: Show results
    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)

    if final_plan:
        print("✅ Planning successful!")
        print(f"📋 Final plan: {len(final_plan.stages)} stages")
        print(f"🎯 Final key: {final_plan.final_key}")
        print(f"🔄 Iterations: {len(history)}")

        print("\n📈 Quality progression:")
        for i, iteration in enumerate(history, 1):
            score = planner._extract_score_from_feedback(iteration["feedback"])
            print(f"  Iteration {i}: {score}/10 ({iteration['num_stages']} stages)")

        print("\n📝 Stage Overview:")
        for i, stage in enumerate(final_plan.stages, 1):
            print(f"  {i}. {stage.id}")
            print(f"     Reads: {stage.reads}")
            print(f"     Writes: {stage.writes}")

        # Save detailed results
        output_file = Path("enhanced_pipeline_results.md")
        with open(output_file, 'w') as f:
            f.write("# Enhanced Pipeline Test Results\n\n")
            f.write(f"**Algorithm:** {algorithm}\n\n")
            f.write("## Extracted Knowledge\n\n")
            f.write(knowledge)
            f.write("\n\n## Generated Plan\n\n")
            f.write(f"**Stages:** {len(final_plan.stages)}\n")
            f.write(f"**Final Key:** {final_plan.final_key}\n\n")

            for i, stage in enumerate(final_plan.stages, 1):
                f.write(f"### Stage {i}: {stage.id}\n")
                f.write(f"- **Reads:** {stage.reads}\n")
                f.write(f"- **Writes:** {stage.writes}\n")
                f.write(f"- **Prompt:** {stage.prompt_template[:100]}...\n\n")

            f.write("## Quality History\n\n")
            for i, iteration in enumerate(history, 1):
                score = planner._extract_score_from_feedback(iteration["feedback"])
                f.write(f"**Iteration {i}:** {score}/10\n\n")
                f.write("Feedback:\n```\n")
                f.write(iteration["feedback"][:500] + "..." if len(iteration["feedback"]) > 500 else iteration["feedback"])
                f.write("\n```\n\n")

        print(f"\n💾 Detailed results saved to: {output_file}")
        print("✅ Enhanced pipeline test completed successfully!")

    else:
        print("❌ Planning failed")


if __name__ == "__main__":
    asyncio.run(test_full_enhanced_pipeline())