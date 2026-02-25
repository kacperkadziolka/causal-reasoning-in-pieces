import re
from typing import Any, Optional


def parse_graph_from_question(question: str) -> Optional[dict]:
    """Parse graph structure from the NLGraph question text for validation."""
    try:
        # Extract node range
        node_match = re.search(r'nodes are numbered from (\d+) to (\d+)', question)
        if not node_match:
            return None
        start_node = int(node_match.group(1))
        end_node = int(node_match.group(2))
        nodes = list(range(start_node, end_node + 1))

        # Extract edges
        edge_pattern = r'edge between node (\d+) and node (\d+) with weight (\d+)'
        edges = {}
        for match in re.finditer(edge_pattern, question):
            u, v, w = int(match.group(1)), int(match.group(2)), int(match.group(3))
            edges.setdefault(u, {})[v] = w
            edges.setdefault(v, {})[u] = w

        # Extract source and target
        path_match = re.search(r'shortest path from node (\d+) to node (\d+)', question)
        source = int(path_match.group(1)) if path_match else None
        target = int(path_match.group(2)) if path_match else None

        return {
            "nodes": nodes,
            "edges": edges,
            "source": source,
            "target": target,
        }
    except Exception:
        return None


def validate_path(path: list[int], graph_edges: dict, source: int, target: int) -> dict:
    """
    Validate a path against the graph structure.
    Returns dict with 'valid', 'computed_weight', 'errors'.
    """
    errors = []

    if not path:
        return {"valid": False, "computed_weight": None, "errors": ["Empty path"]}

    if path[0] != source:
        errors.append(f"Path starts at {path[0]}, expected {source}")
    if path[-1] != target:
        errors.append(f"Path ends at {path[-1]}, expected {target}")

    computed_weight = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u in graph_edges and v in graph_edges[u]:
            computed_weight += graph_edges[u][v]
        else:
            errors.append(f"No edge between {u} and {v}")
            computed_weight = None
            break

    return {
        "valid": len(errors) == 0,
        "computed_weight": computed_weight,
        "errors": errors,
    }


def compare_shortest_path(result: dict, ground_truth_weight: int, question: str) -> dict:
    """Compare model result against ground truth."""
    model_path = result.get("path")
    model_weight = result.get("total_weight")

    comparison = {
        "weight_correct": False,
        "path_valid": False,
        "path_weight_consistent": False,
        "extraction_failed": False,
    }

    # Check extraction
    if model_path is None and model_weight is None:
        comparison["extraction_failed"] = True
        return comparison

    # Check weight
    if model_weight is not None:
        comparison["weight_correct"] = (model_weight == ground_truth_weight)

    # Validate path against graph
    graph_info = parse_graph_from_question(question)
    if graph_info and model_path:
        validation = validate_path(
            model_path,
            graph_info["edges"],
            graph_info["source"],
            graph_info["target"],
        )
        comparison["path_valid"] = validation["valid"]

        # Cross-check: does computed path weight match claimed weight?
        if validation["computed_weight"] is not None and model_weight is not None:
            comparison["path_weight_consistent"] = (validation["computed_weight"] == model_weight)

    return comparison


def aggregate_metrics(results: list[dict], difficulties: list[str]) -> dict:
    """Aggregate evaluation metrics with difficulty breakdown."""
    total = len(results)
    if total == 0:
        return {"error": "No results to aggregate"}

    weight_correct = sum(1 for r in results if r["weight_correct"])
    path_valid = sum(1 for r in results if r["path_valid"])
    extraction_failed = sum(1 for r in results if r["extraction_failed"])
    consistent = sum(1 for r in results if r["path_weight_consistent"])

    metrics = {
        "total": total,
        "weight_accuracy": weight_correct / total,
        "path_validity_rate": path_valid / total,
        "extraction_failure_rate": extraction_failed / total,
        "path_weight_consistency": consistent / total,
        "by_difficulty": {},
    }

    # Breakdown by difficulty
    difficulty_set = sorted(set(difficulties))
    for diff in difficulty_set:
        indices = [i for i, d in enumerate(difficulties) if d == diff]
        diff_results = [results[i] for i in indices]
        diff_total = len(diff_results)
        if diff_total == 0:
            continue
        diff_correct = sum(1 for r in diff_results if r["weight_correct"])
        diff_failed = sum(1 for r in diff_results if r["extraction_failed"])
        metrics["by_difficulty"][diff] = {
            "total": diff_total,
            "weight_accuracy": diff_correct / diff_total,
            "extraction_failures": diff_failed,
        }

    return metrics


def display_metrics(metrics: dict) -> None:
    """Display evaluation metrics in a formatted way."""
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print("\n" + "=" * 50)
    print("SHORTEST PATH EVALUATION METRICS")
    print("=" * 50)
    print(f"Total samples:          {metrics['total']}")
    print(f"Weight accuracy:        {metrics['weight_accuracy']:.2%}")
    print(f"Path validity rate:     {metrics['path_validity_rate']:.2%}")
    print(f"Path-weight consistency: {metrics['path_weight_consistency']:.2%}")
    print(f"Extraction failure rate: {metrics['extraction_failure_rate']:.2%}")

    if metrics["by_difficulty"]:
        print("\nBy Difficulty:")
        print("-" * 40)
        for diff, diff_metrics in metrics["by_difficulty"].items():
            print(f"  {diff.capitalize()}: {diff_metrics['weight_accuracy']:.2%} accuracy "
                  f"({diff_metrics['total']} samples, "
                  f"{diff_metrics['extraction_failures']} extraction failures)")

    print("=" * 50)
