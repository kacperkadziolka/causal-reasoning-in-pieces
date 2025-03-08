import json
import re


def extract_premise(text: str) -> str:
    """
    Extracts Premise from the input text.

    Parameters:
    - text (str): The input text containing "Premise:" and possibly "Hypothesis:".

    Returns:
    - str: The extracted premise.
    """
    # Use regex to extract text between "Premise:" and "Hypothesis:"
    match = re.search(r"Premise:\s*(.*?)\s*Hypothesis:", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If "Hypothesis:" is not present, extract everything after "Premise:"
        match = re.search(r"Premise:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # If "Premise:" is not found, return the original text
            return text.strip()


def compare_edges(expected_edges: set, answer_edges: set) -> dict:
    """
    Compare the expected edges with the model's predicted edges.

    :param expected_edges: The ground-truth edges.
    :param answer_edges: The model's predicted edges.
    :return: A dictionary with metrics for comparison.
    """
    # Check if the edges are an exact match
    exact_match = expected_edges == answer_edges

    if exact_match:
        print("The model's predicted edges match the expected edges!")
    else:
        print("Mismatch found:")
        print(f"Expected edges: {expected_edges}")
        print(f"Model edges: {answer_edges}")
        print(f"Missing in model: {expected_edges - answer_edges}")
        print(f"Extra in model: {answer_edges - expected_edges}")

    # Calculate detailed metrics
    true_positive = len(expected_edges & answer_edges)
    false_positive = len(answer_edges - expected_edges)
    false_negative = len(expected_edges - answer_edges)

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "expected_count": len(expected_edges),
        "predicted_count": len(answer_edges),
        "exact_match": exact_match,
        "missing_edges": list(expected_edges - answer_edges),
        "extra_edges": list(answer_edges - expected_edges)
    }


def extract_edges_incident_format(answer: str, step: int) -> set:
    """
    Extract the causal edges from the provided LLM answer string using the adjacency list format.

    :param answer: The answer returned by the LLM API.
    :return: A set of edges extracted from the answer, represented as sorted tuples.
    """
    if step == 4:
        step_pattern = r"Step 4: Compile the Causal Undirected Skeleton"
    elif step == 5:
        step_pattern = r"Step 5: Compile the Causal Undirected Skeleton"
    else:
        raise ValueError("Step number not recognized.")

    try:
        # Normalize line breaks
        answer = answer.replace("\r\n", "\n").replace("\r", "\n")

        # Locate step with answer
        step_match = re.search(step_pattern, answer, flags=re.IGNORECASE)
        if not step_match:
            raise ValueError("Step 5 section not found in the answer.")

        """
        Below code works well with the few-shot example, where the answer are more consistent with
        requested format. In zero-shot settings, model often fails to includes the "In this graph:" 
        line in the output.
        """
        # Find the start of the adjacency list
        adjacency_start = answer.find("In this graph:", step_match.end())
        if adjacency_start == -1:
            raise ValueError("Adjacency list section not found in Step 5.")

        # Split lines from the adjacency list
        adjacency_section = answer[adjacency_start:].splitlines()

        edges = set()

        # Pattern for lines:
        #   - Node X has no connections.
        no_conn_pattern = re.compile(r"^- Node\s+(\w+)\s+has\s+no\s+connections\.$", re.IGNORECASE)

        # Pattern to match lines:
        #   - Node A is connected to node C.
        #   - Node A is connected only to node D.
        #   - Node B is connected to nodes A, C and D.
        #   - Node C is connected to node A and node B.
        conn_pattern = re.compile(
            # r"^- Node\s+(\w+)\s+is\s+connected(?:\s+only)?\s+to\s+nodes?\s+(.+)\.$",
            r"^- Node\s+(\w+)\s+is\s+connected(?:\s+only)?\s+to\s+nodes?\s+(.+?)\.?$",
            re.IGNORECASE
        )

        for line in adjacency_section:
            line = line.strip()

            if not line.startswith("- Node"):
                continue  # skip lines that don't describe adjacency

            # Case 1: "... has no connections."
            no_conn_match = no_conn_pattern.match(line)
            if no_conn_match:
                continue

            # Case 2: "... is connected to node(s) ..."
            node_conn_match = conn_pattern.match(line)
            if node_conn_match:
                node = node_conn_match.group(1)
                connections_str = node_conn_match.group(2)

                # Remove "node " from connections (handles "node A and node B")
                connections_str = re.sub(r"\bnode\s+", "", connections_str, flags=re.IGNORECASE)

                # Split on commas or 'and'
                connections = re.split(r"(?:,\s*|and\s+)", connections_str)
                connections = [c.strip() for c in connections if c.strip()]

                # Build edges
                for conn in connections:
                    edge = tuple(sorted([node, conn]))
                    edges.add(edge)

        return edges

    except Exception as e:
        raise RuntimeError(f"Failed to extract edges: {e}")


def aggregate_metrics(results: list) -> dict:
    """
    Aggregate metrics from multiple experiments.

    :param results: A list of comparison results from multiple experiments.
    :return: A dictionary with aggregated metrics.
    """
    total_true_positive = sum(r["true_positive"] for r in results)
    total_false_positive = sum(r["false_positive"] for r in results)
    total_false_negative = sum(r["false_negative"] for r in results)
    total_expected_count = sum(r["expected_count"] for r in results)
    total_exact_matches = sum(1 for r in results if r["exact_match"])
    total_prompts = len(results)

    precision = total_true_positive / (total_true_positive + total_false_positive) if (total_true_positive + total_false_positive) > 0 else 0
    recall = total_true_positive / (total_true_positive + total_false_negative) if (total_true_positive + total_false_negative) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_true_positive / total_expected_count if total_expected_count > 0 else 0
    exact_match_ratio = total_exact_matches / total_prompts if total_prompts > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "exact_match_ratio": exact_match_ratio,
        "total_exact_matches": total_exact_matches,
        "total_prompts": total_prompts,
    }


def display_metrics(metrics: dict) -> None:
    """
    Display all the metrics in a user-friendly format.

    :param metrics: Dictionary of aggregated metrics.
    """
    print("\nAggregate Metrics:")
    for key, value in metrics.items():
        # Format floating-point numbers with 2 decimal places
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').capitalize()}: {value}")


def extract_edges_json(answer: str) -> set:
    """
    Extract the causal edges from the provided LLM answer string using the JSON format.

    :param answer: The answer returned by the LLM API.
    :return: A set of edges extracted from the answer, represented as sorted tuples.
    """
    try:
        # First approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)

            # Convert edges to set of sorted tuples
            if "edges" in data:
                edges = set(tuple(sorted(edge)) for edge in data["edges"])
                return edges

        # Second approach: Find all potential JSON objects and test each one
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "edges" in data:
                    edges = set(tuple(sorted(edge)) for edge in data["edges"])
                    return edges
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON with edges found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract edges: {e}")
