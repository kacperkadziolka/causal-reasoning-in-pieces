import asyncio
import argparse
from typing import Any, Optional, Dict
from dotenv import load_dotenv
import pandas as pd

from knowledge.extractor import EnhancedKnowledgeExtractor
from plan.iterative_planner import IterativePlanner
from plan.multi_agent_planner import MultiAgentPlanner
from plan.models import Plan
from execute.executor import run_plan
from validate.validator_generator import GenericValidatorGenerator
from utils.logging_config import init_logger
from tasks import get_task
from tasks.base import BaseTask

load_dotenv()


async def run_simple_workflow(
    sample: pd.Series,
    task: BaseTask,
    use_sequential_generation: bool = False,
    use_multi_agent_planner: bool = False,
    cached_plan: Optional[Plan] = None,
    cached_knowledge: Optional[str] = None,
    cached_structured_constraints: Optional[Dict] = None,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run the main workflow: knowledge extraction -> planning -> execution.

    Args:
        sample: A pandas Series containing the test sample data
        task: Task configuration that provides algorithm info and evaluation logic
        use_sequential_generation: If True, generate stage prompts sequentially
                                   (only used with multi-agent planner)
        use_multi_agent_planner: If True, use MultiAgentPlanner instead of IterativePlanner
        cached_plan: Optional pre-generated plan to skip planning phase
        cached_knowledge: Optional pre-extracted knowledge to skip extraction phase
        cached_structured_constraints: Optional pre-extracted constraints for validation
        verbose: If True, show detailed execution logs; if False, show minimal output

    Returns:
        Dictionary with execution results or None if failed
    """

    if verbose:
        print("\n🚀 WORKFLOW")
        print("=" * 60)

    # STEP 1: Knowledge Extraction (skip if cached)
    if cached_knowledge is not None:
        if verbose:
            print("📦 Using cached knowledge")
        knowledge = cached_knowledge
        structured_constraints = cached_structured_constraints
    else:
        if verbose:
            print("\n📚 STEP 1: Knowledge Extraction")
            print("-" * 30)
        extractor = EnhancedKnowledgeExtractor()
        knowledge, structured_constraints = await extractor.extract_simple_knowledge(task.algorithm_name, sample["input"])
        if verbose:
            print(f"✅ Knowledge extracted: {knowledge}")

    # STEP 2: Planning (skip if cached)
    if cached_plan is not None:
        if verbose:
            print("📦 Using cached plan")
        plan = cached_plan
    else:
        if verbose:
            print("\n🔄 STEP 2: Planning")
            print("-" * 30)

        if use_multi_agent_planner:
            if verbose:
                print("Using MultiAgentPlanner" + (" (SEQUENTIAL mode)" if use_sequential_generation else " (BATCH mode)"))
            planner = MultiAgentPlanner()
            plan, metadata = await planner.generate_plan(
                task_description=task.task_description,
                algorithm_knowledge=knowledge,
                use_sequential=use_sequential_generation
            )
            if verbose:
                print(f"✅ Planning successful: {len(plan.stages)} stages")
                print(f"   Generation mode: {metadata.get('generation_mode', 'unknown')}")
        else:
            if verbose:
                print("Using IterativePlanner (two-stage planning)")
            planner = IterativePlanner()
            plan = await planner.generate_two_stage_plan(
                task_description=task.task_description,
                algorithm_knowledge=knowledge,
                enhance_prompts=True
            )
            if verbose:
                print(f"✅ Planning successful: {len(plan.stages)} stages")

    if verbose:
        print("\n⚡ STEP 3: Execution")
        print("-" * 30)

    # Create validator from structured constraints (if available)
    validator = None
    if structured_constraints and structured_constraints.get("stage_constraints"):
        validator = GenericValidatorGenerator(structured_constraints)
        if verbose:
            print(f"   🔒 Validator created with {len(structured_constraints.get('stage_constraints', {}))} stage constraints")

    initial_context = {"input": sample["input"]}
    final_context = await run_plan(plan, initial_context, verbose=verbose, validator=validator)
    final_key = plan.final_key or "result"
    final_result = final_context.get(final_key)

    if verbose:
        print("✅ Execution completed")
        print(f"🎯 Final result key: '{final_key}'")
        print(f"📊 Final result: {final_result}")

        print("\n📊 STEP 4: Evaluation")
        print("-" * 30)

    # Task-specific result extraction and evaluation
    try:
        extracted = task.extract_result(final_result, final_key, plan)
    except ValueError as e:
        print(f"❌ Result extraction failed: {e}")
        print(f"🔍 Final result: {final_result}")
        raise

    eval_result = task.evaluate(extracted, sample)
    is_correct = eval_result["is_correct"]

    if verbose:
        print(f"🎯 Predicted: {eval_result['predicted_summary']}")
        print(f"📊 Expected: {eval_result['expected_summary']}")
        print(f"✅ Result: {'CORRECT' if is_correct else 'INCORRECT'}")

    return {
        "sample_idx": sample.name,
        "knowledge_length": len(knowledge),
        "num_stages": len(plan.stages),
        "is_correct": is_correct,
        "final_context": final_context,
        **eval_result,
    }


async def main(
    task_name: str = "causal_discovery",
    sample_idx: Optional[int] = None,
    use_sequential_generation: bool = False,
    use_multi_agent_planner: bool = False,
):
    """
    Main entry point for running the pipeline on a single sample.

    Args:
        task_name: Which task to run ("causal_discovery" or "shortest_path")
        sample_idx: Specific sample index or None for random
        use_sequential_generation: Enable sequential stage generation
        use_multi_agent_planner: Use MultiAgentPlanner instead of IterativePlanner
    """
    task = get_task(task_name)

    print("🔬 SELF-PLANNED PIPELINE")
    print("=" * 60)
    print(f"📋 Task: {task_name} ({task.algorithm_name})")

    if use_multi_agent_planner:
        print(f"🧠 Planner: MultiAgentPlanner ({'SEQUENTIAL' if use_sequential_generation else 'BATCH'} mode)")
    else:
        print("🧠 Planner: IterativePlanner (two-stage)")
    print("=" * 60)

    # Load dataset and fetch sample
    dataset = task.load_dataset(task.default_dataset_path)
    sample = task.fetch_sample(dataset, sample_idx)

    # Run workflow
    result = await run_simple_workflow(
        sample,
        task=task,
        use_sequential_generation=use_sequential_generation,
        use_multi_agent_planner=use_multi_agent_planner
    )

    # Final summary
    print("\n" + "=" * 60)
    print("📋 PIPELINE SUMMARY")
    print("=" * 60)

    if result:
        print("✅ Workflow completed successfully!")
        print(f"📚 Knowledge: {result['knowledge_length']} chars")
        print(f"📝 Stages: {result['num_stages']}")
        print(f"🎯 Predicted: {result['predicted_summary']}")
        print(f"📊 Expected: {result['expected_summary']}")
        print(f"📊 Result: {'✅ CORRECT' if result['is_correct'] else '❌ INCORRECT'}")
    else:
        print("❌ Workflow failed")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run self-planned pipeline on specific dataset samples")
    parser.add_argument(
        "--task",
        type=str,
        default="causal_discovery",
        choices=["causal_discovery", "shortest_path"],
        help="Which task to run (default: causal_discovery)"
    )
    parser.add_argument("--sample-idx", type=int, default=110, help="Specific sample index to test")
    parser.add_argument(
        "--sequential-generation",
        action="store_true",
        default=True,
        help="Generate stage prompts sequentially (only with --multi-agent-planner)"
    )
    parser.add_argument(
        "--multi-agent-planner",
        action="store_true",
        default=True,
        help="Use MultiAgentPlanner instead of IterativePlanner"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode with verbose logging"
    )

    args = parser.parse_args()

    # Initialize logger with debug setting
    init_logger(debug=args.debug)

    print(f"📋 Task: {args.task}")

    if args.sample_idx is not None:
        print(f"🎯 Testing sample index: {args.sample_idx}")
    else:
        print("🎯 Using random sample")

    if args.debug:
        print("🐛 Debug mode enabled - showing verbose logs")

    asyncio.run(main(
        task_name=args.task,
        sample_idx=args.sample_idx,
        use_sequential_generation=args.sequential_generation,
        use_multi_agent_planner=args.multi_agent_planner
    ))
