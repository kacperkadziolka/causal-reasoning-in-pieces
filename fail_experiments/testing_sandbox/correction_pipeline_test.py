import ast

import pandas as pd

from fail_experiments.testing_sandbox.conf_groq import get_test_model
from fail_experiments.testing_sandbox.correction_pipeline import CorrectionPipeline


def verify_edge_test(pipeline: CorrectionPipeline, df: pd.DataFrame, num_samples: int) -> None:
    """
    Test the verify edges of correction pipeline and print statistics.

    Args:
        pipeline: The correction pipeline object
        df: The dataset to test on
        num_samples: Number of examples to test
    """
    # Convert string representations to lists/tuples
    for col in ['missing_edges', 'extra_edges', 'expected_edges']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)

    samples = df.sample(num_samples)
    stats = {
        'missing_correct': 0,
        'missing_total': 0,
        'extra_correct': 0,
        'extra_total': 0,
        'hallucination_correct': 0,
        'hallucination_total': 0
    }

    for idx, example in samples.iterrows():
        premise = example['premise']
        print(f"\n===== Testing example {idx} =====")

        # Test missing edges
        for edge in example['missing_edges']:
            stats['missing_total'] += 1
            should_exist = pipeline.verify_edge(premise, edge)
            print(f"Missing edge {edge}: Model says {'should exist' if should_exist else 'should not exist'}")
            if should_exist:
                stats['missing_correct'] += 1

        # Test extra edges
        for edge in example['extra_edges']:
            stats['extra_total'] += 1
            should_exist = pipeline.verify_edge(premise, edge)
            print(f"Extra edge {edge}: Model says {'should exist' if should_exist else 'should not exist'}")
            if not should_exist:
                stats['extra_correct'] += 1

        # Test hallucination on some correct edges
        correct_edges = [e for e in example['expected_edges']
                        if e not in example['missing_edges'] and e not in example['extra_edges']]

        for edge in correct_edges:
            stats['hallucination_total'] += 1
            should_exist = pipeline.verify_edge(premise, edge)
            print(f"Correct edge {edge}: Model says {'should exist' if should_exist else 'should not exist'}")
            if should_exist:
                stats['hallucination_correct'] += 1

    # Print overall statistics
    print("\n===== Overall Statistics =====")

    # Fix the division by zero errors with proper checks
    missing_pct = (stats['missing_correct'] / stats['missing_total'] * 100) if stats['missing_total'] > 0 else 0
    print(f"Missing edges: {stats['missing_correct']}/{stats['missing_total']} correctly identified "
          f"({missing_pct:.1f}%)")

    extra_pct = (stats['extra_correct'] / stats['extra_total'] * 100) if stats['extra_total'] > 0 else 0
    print(f"Extra edges: {stats['extra_correct']}/{stats['extra_total']} correctly identified "
          f"({extra_pct:.1f}%)")

    hallucination_pct = (stats['hallucination_correct'] / stats['hallucination_total'] * 100) if stats[
                                                                                                     'hallucination_total'] > 0 else 0
    print(
        f"Hallucination test: {stats['hallucination_correct']}/{stats['hallucination_total']} correct edges preserved "
        f"({hallucination_pct:.1f}%)")

    # Calculate overall score
    total_cases = stats['missing_total'] + stats['extra_total'] + stats['hallucination_total']
    total_correct = stats['missing_correct'] + stats['extra_correct'] + stats['hallucination_correct']

    if total_cases > 0:
        print(f"Overall accuracy: {total_correct}/{total_cases} ({total_correct / total_cases * 100:.1f}%)")
    else:
        print("Overall accuracy: No cases to evaluate")


def global_consistency_test(pipeline: CorrectionPipeline, df: pd.DataFrame, num_samples: int) -> None:
    """
    Test the global consistency check of correction pipeline and print statistics.

    Args:
        pipeline: The correction pipeline object
        df: The dataset to test on
        num_samples: Number of examples to test
    """
    for col in ['model_edges', 'missing_edges', 'extra_edges', 'expected_edges']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)

    samples = df.sample(num_samples)
    stats = {
        'initial_missing': 0,  # Missing edges in model's edges
        'final_missing': 0,  # Missing edges after correction
        'initial_extra': 0,  # Extra edges in model's edges
        'final_extra': 0,  # Extra edges after correction
        'total_examples': 0  # Number of examples processed
    }

    for idx, example in samples.iterrows():
        premise = example['premise']
        expected_edges = example['expected_edges']
        missing_edges = example['missing_edges']
        extra_edges = example['extra_edges']
        model_edges = example['model_edges']

        print(f"\n===== Testing global consistency for example {idx} =====")
        print(f"Expected edges: {expected_edges}")
        print(f"Model edges: {model_edges}")
        print(f"Missing edges: {missing_edges}")
        print(f"Extra edges: {extra_edges}")

        # Run global consistency check
        corrected_edges = pipeline.check_global_consistency(premise, model_edges)


def main():
    df: pd.DataFrame = pd.read_csv("failed_experiments_premise.csv")
    pipeline: CorrectionPipeline = CorrectionPipeline(get_test_model())

    # Test verify edge
    # print("\n===== TESTING EDGE VERIFICATION =====")
    # verify_edge_test(pipeline, df, 1)

    # Test global consistency
    print("\n===== TESTING GLOBAL CONSISTENCY =====")
    global_consistency_test(pipeline, df, 1)


if __name__ == "__main__":
    main()
