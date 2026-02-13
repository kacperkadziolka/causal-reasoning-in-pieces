import json
import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from tasks.base import BaseTask


class ShortestPathTask(BaseTask):
    """Task configuration for shortest path on the NLGraph dataset."""

    algorithm_name = "Dijkstra's Algorithm"

    task_description = """
Task: Given a natural-language description of a weighted undirected graph and a source-target query, find the shortest (minimum total weight) path from the source node to the target node.

- The input describes: nodes numbered 0 to N, undirected edges with integer weights, and a query "find the shortest path from node X to node Y".
- Apply Dijkstra's algorithm (or equivalent shortest-path method) to find the optimal path.
- The path must be a sequence of connected nodes from source to target where each consecutive pair shares an edge in the graph.
- The total weight is the sum of all edge weights along the path.

ENVIRONMENT:
- All graph information is given EXPLICITLY in the input as natural language text.
- Parse the graph carefully: extract ALL nodes, ALL edges, and ALL weights.
- The graph is UNDIRECTED — edges can be traversed in either direction.
- Do NOT invent edges or weights that are not in the input.

CRITICAL EXECUTION CONSTRAINT — PLAN STRUCTURE:
- The execution pipeline runs each stage EXACTLY ONCE in sequence. There is NO loop or iteration mechanism between stages.
- Therefore, Dijkstra's main loop (select min-distance vertex, relax neighbors, mark visited, repeat) MUST be executed ENTIRELY WITHIN A SINGLE STAGE.
- Do NOT split the iterative loop across multiple stages — that will fail because each stage runs only once.
- Recommended plan structure (3 stages):
  1. **Parse input**: Extract graph structure (nodes, edges with weights) and identify source/target from the natural language text. Output a structured graph representation.
  2. **Run Dijkstra's algorithm**: Take the parsed graph, source, and target. Execute the COMPLETE algorithm within this single stage: initialize distances, run the full relaxation loop until termination, then reconstruct the shortest path from source to target using the predecessor array.
  3. **Format output**: Take the computed path and total weight and format them as the final JSON output.
- You MAY combine stages 2 and 3 into a single stage if preferred, but do NOT split the iterative algorithm across separate stages.

ALGORITHM DETAILS (for reference within the execution stage):
- Initialization: set distance to source = 0, all others = infinity.
- Main loop: repeatedly pick the unvisited node with smallest tentative distance, relax all its neighbors (update distance if shorter path found), mark as visited.
- Termination: when target is visited or all reachable nodes are processed.
- Path reconstruction: trace back from target to source using the predecessor array.

Input available in context: "input" (contains natural language graph description with edges, weights, and source-target query).

CRITICAL OUTPUT FORMAT:
- The final stage MUST output a JSON with two keys:
  - "path": array of integers representing the shortest path nodes (e.g., [0, 3, 5, 2])
  - "total_weight": integer representing the total weight of the shortest path (e.g., 7)
- Example: {"path": [0, 3, 5, 2], "total_weight": 7}
- "path" MUST be an array of integers, NOT a boolean.
- "total_weight" MUST be an integer, NOT a boolean.
- DO NOT include extra fields or nested objects.
"""

    default_dataset_path = "../../data/nlgraph_shortest_path_main.json"

    def load_dataset(self, path: str) -> pd.DataFrame:
        with open(path, "r") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            samples = []
            for idx, entry in raw.items():
                sample = {"id": int(idx), **entry}
                if "num_nodes" not in sample:
                    sample.update(_parse_graph_metadata(sample["question"], sample["answer"]))
                samples.append(sample)
            samples.sort(key=lambda x: x["id"])
        elif isinstance(raw, list):
            samples = raw
        else:
            raise ValueError(f"Unexpected JSON format: {type(raw)}")

        for sample in samples:
            sample["input"] = sample["question"]

        df = pd.DataFrame(samples)
        print(f"Dataset loaded: {len(df)} samples")
        return df

    def fetch_sample(self, dataset: pd.DataFrame, sample_idx: Optional[int] = None) -> pd.Series:
        if sample_idx is not None:
            if sample_idx < 0 or sample_idx >= len(dataset):
                raise ValueError(f"Sample index {sample_idx} out of range (0-{len(dataset)-1})")
            actual_idx = sample_idx
            print(f"Using specified index: {actual_idx}")
        else:
            actual_idx = random.randint(0, len(dataset) - 1)
            print(f"Using random index: {actual_idx}")

        sample = dataset.iloc[actual_idx]

        print(f"Index: {actual_idx}")
        print(f"Difficulty: {sample['difficulty']}")
        print(f"Nodes: {sample.get('num_nodes', '?')}, Edges: {sample.get('num_edges', '?')}")
        print(f"Ground truth weight: {_ensure_ground_truth_weight(sample)}")
        print(f"Question: {sample['question'][:200]}...")
        print("=" * 50)

        return sample

    def extract_result(
        self,
        final_result: Any,
        final_key: str,
        plan: Any,
    ) -> Dict[str, Any]:
        """Extract path and weight from the pipeline's final output."""
        if final_result is None:
            raise ValueError("Final result is None — pipeline produced no output")

        model_path = None
        model_weight = None

        if isinstance(final_result, dict):
            # Look for path in common key names
            for key in ["path", "shortest_path", "route"]:
                if key in final_result:
                    val = final_result[key]
                    if isinstance(val, list):
                        try:
                            model_path = [int(n) for n in val]
                        except (ValueError, TypeError) as e:
                            print(f"WARNING: Cannot parse path as integers: {val} ({e})")
                    break

            # Look for weight in common key names
            for key in ["total_weight", "weight", "distance", "cost", "total_distance"]:
                if key in final_result:
                    val = final_result[key]
                    if val is not None:
                        try:
                            model_weight = int(val)
                        except (ValueError, TypeError) as e:
                            print(f"WARNING: Cannot parse weight as integer: {val} ({e})")
                    break
        elif isinstance(final_result, list):
            try:
                model_path = [int(n) for n in final_result]
            except (ValueError, TypeError) as e:
                print(f"WARNING: Cannot parse path list as integers: {final_result} ({e})")
        elif isinstance(final_result, (int, float)):
            model_weight = int(final_result)

        if model_path is None and model_weight is None:
            print(f"WARNING: Could not extract path or weight from final result: {final_result}")

        return {"model_path": model_path, "model_weight": model_weight}

    def evaluate(
        self,
        extracted: Dict[str, Any],
        sample: pd.Series,
    ) -> Dict[str, Any]:
        ground_truth_weight = _ensure_ground_truth_weight(sample)
        model_path = extracted.get("model_path")
        model_weight = extracted.get("model_weight")

        weight_correct = (model_weight == ground_truth_weight) if model_weight is not None else False
        extraction_failed = model_weight is None

        # Path validation
        path_valid = None
        if model_path:
            graph_info = _parse_graph_from_question(sample["question"])
            if graph_info["edges"]:
                is_valid, computed_weight = _validate_path(
                    model_path, graph_info["edges"], graph_info["source"], graph_info["target"]
                )
                path_valid = is_valid
                if is_valid and model_weight is not None and computed_weight != model_weight:
                    print(f"WARNING: Model path weight ({computed_weight}) != claimed weight ({model_weight})")

        difficulty = sample.get("difficulty", "unknown")

        return {
            "is_correct": weight_correct,
            "predicted_summary": f"weight={model_weight}, path={model_path}",
            "expected_summary": f"weight={ground_truth_weight}",
            "weight_correct": weight_correct,
            "path_valid": path_valid,
            "extraction_failed": extraction_failed,
            "difficulty": difficulty,
            "model_weight": model_weight,
            "ground_truth_weight": ground_truth_weight,
        }

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [r for r in results if r.get("error") is None]
        if not successful:
            return {"total": 0, "weight_accuracy": 0.0}

        total = len(successful)
        weight_correct = sum(1 for r in successful if r.get("weight_correct", False))
        extraction_failures = sum(1 for r in successful if r.get("extraction_failed", True))

        valid_paths = [r for r in successful if r.get("path_valid") is not None]
        path_valid_count = sum(1 for r in valid_paths if r["path_valid"])

        metrics = {
            "total": total,
            "weight_correct": weight_correct,
            "weight_accuracy": weight_correct / total if total > 0 else 0.0,
            "extraction_failures": extraction_failures,
            "extraction_failure_rate": extraction_failures / total if total > 0 else 0.0,
            "path_valid_rate": path_valid_count / len(valid_paths) if valid_paths else None,
        }

        # Breakdown by difficulty
        by_difficulty = {}
        difficulties = sorted(set(r.get("difficulty") for r in successful if r.get("difficulty")))
        for diff in difficulties:
            diff_results = [r for r in successful if r.get("difficulty") == diff]
            if not diff_results:
                continue
            diff_total = len(diff_results)
            diff_correct = sum(1 for r in diff_results if r.get("weight_correct", False))
            diff_failures = sum(1 for r in diff_results if r.get("extraction_failed", True))
            by_difficulty[diff] = {
                "total": diff_total,
                "weight_correct": diff_correct,
                "weight_accuracy": diff_correct / diff_total if diff_total > 0 else 0.0,
                "extraction_failures": diff_failures,
            }

        metrics["by_difficulty"] = by_difficulty
        return metrics

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        print(f"Weight Accuracy:       {metrics['weight_accuracy']*100:.1f}%")
        print(f"Extraction Failures:   {metrics.get('extraction_failures', 0)} ({metrics.get('extraction_failure_rate', 0)*100:.1f}%)")

        if metrics.get("path_valid_rate") is not None:
            print(f"Path Validity Rate:    {metrics['path_valid_rate']*100:.1f}%")

        if metrics.get("by_difficulty"):
            print("\nBy Difficulty:")
            print("-" * 40)
            for diff, dm in metrics["by_difficulty"].items():
                print(
                    f"  {diff.capitalize():>5}: "
                    f"{dm['weight_accuracy']:.2%} accuracy "
                    f"({dm['total']} samples, "
                    f"{dm['extraction_failures']} extraction failures)"
                )


# --- Helper functions (adapted from shortest_path_baseline/answer_extractor.py & utils.py) ---


def _parse_graph_metadata(question: str, answer: str) -> dict:
    node_match = re.search(r"numbered from (\d+) to (\d+)", question)
    num_nodes = int(node_match.group(2)) + 1 if node_match else 0

    edges = re.findall(r"edge between node (\d+) and node (\d+) with weight (\d+)", question)
    num_edges = len(edges)

    path_match = re.search(r"is ([\d,]+) with a total weight of (\d+)", answer)
    if path_match:
        path = [int(n) for n in path_match.group(1).split(",")]
        total_weight = int(path_match.group(2))
    else:
        path = []
        total_weight = None

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "path_length": len(path),
        "total_weight": total_weight,
    }


def _ensure_ground_truth_weight(sample) -> int:
    weight = sample.get("total_weight")
    if weight is not None and not (isinstance(weight, float) and pd.isna(weight)):
        return int(weight)

    match = re.search(r"total weight of (\d+)", sample.get("answer", ""))
    if match:
        return int(match.group(1))

    raise ValueError(f"Cannot determine ground truth weight for sample {sample.get('id')}")


def _parse_graph_from_question(question: str) -> dict:
    result = {"num_nodes": 0, "edges": [], "source": None, "target": None}

    node_match = re.search(r"numbered from (\d+) to (\d+)", question)
    if node_match:
        result["num_nodes"] = int(node_match.group(2)) + 1

    edge_pattern = r"edge between node (\d+) and node (\d+) with weight (\d+)"
    for match in re.finditer(edge_pattern, question):
        node1, node2, weight = int(match.group(1)), int(match.group(2)), int(match.group(3))
        result["edges"].append((node1, node2, weight))

    query_match = re.search(r"shortest path from node (\d+) to node (\d+)", question)
    if query_match:
        result["source"] = int(query_match.group(1))
        result["target"] = int(query_match.group(2))

    return result


def _validate_path(
    path: list,
    edges: list,
    source: int,
    target: int,
) -> tuple:
    if not path or len(path) < 2:
        return False, None

    if path[0] != source or path[-1] != target:
        return False, None

    adj = {}
    for u, v, w in edges:
        adj[(u, v)] = w
        adj[(v, u)] = w

    total_weight = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        if edge not in adj:
            return False, None
        total_weight += adj[edge]

    return True, total_weight
