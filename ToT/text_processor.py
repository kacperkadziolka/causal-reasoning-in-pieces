import re

from ToT.utils import config


def extract_edges(answer: str) -> set[tuple]:
    markers = config.get("markers")
    answer = answer.replace("\r\n", "\n").replace("\r", "\n")

    start_index = None
    for marker in markers:
        match = re.search(re.escape(marker), answer, flags=re.IGNORECASE)
        if match:
            start_index = match.end()
            break

    if start_index is None:
        raise ValueError("None of the expected graph markers were found in the answer. Failed to extract edges.")

    adjacency_section = answer[start_index:].splitlines()
    edges = set()

    # Regex for lines:
    #   - Node X has no connections.
    no_conn_pattern = re.compile(
        r"^- Node\s+(\w+)\s+has\s+no\s+connections\.$",
        re.IGNORECASE
    )

    # Regex for lines:
    #   - Node A is connected to node C.
    #   - Node A is connected only to node D.
    #   - Node B is connected to nodes A, C and D.
    #   - Node C is connected to node A and node B.
    conn_pattern = re.compile(
        r"^- Node\s+(\w+)\s+is\s+connected(?:\s+only)?\s+to\s+nodes?\s+(.+?)\.?$",
        re.IGNORECASE
    )

    for line in adjacency_section:
        line = line.strip()

        # Skip line that does not start with "- Node"
        if not line.startswith("- Node"):
            continue

        # Case 1: Node with no connections
        if no_conn_pattern.match(line):
            continue

        # Case 2: Node with connections
        conn_match = conn_pattern.match(line)
        if conn_match:
            node = conn_match.group(1)
            connections_str = conn_match.group(2)

            # Remove occurrences of "node" (e.g., "node A" becomes "A")
            connections_str = re.sub(r"\bnode\s+", "", connections_str, flags=re.IGNORECASE)

            # Split the connections string on commas or the word "and"
            connections = re.split(r",\s*|and\s+", connections_str)
            connections = [conn.strip() for conn in connections if conn.strip()]

            for conn in connections:
                edge = tuple(sorted([node, conn]))
                edges.add(edge)

    return edges


def compare_edges(expected_edges: set, generated_edges: set) -> dict[str, any]:
    exact_match = expected_edges == generated_edges

    if exact_match:
        print("The model's predicted edges match the expected edges!")
    else:
        print("Mismatch found:")
        print(f"Expected edges: {expected_edges}")
        print(f"Model edges: {generated_edges}")
        print(f"Missing in model: {expected_edges - generated_edges}")
        print(f"Extra in model: {generated_edges - expected_edges}")

    true_positive = len(expected_edges & generated_edges)  # Correctly predicted edges
    false_positive = len(generated_edges - expected_edges)  # Extra edges in prediction
    false_negative = len(expected_edges - generated_edges)  # Missing edges in prediction

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "expected_count": len(expected_edges),
        "predicted_count": len(generated_edges),
        "exact_match": exact_match,
    }


def aggregate_metrics(results: list) -> dict[str, any]:
    total_exact_matches = sum(1 for r in results if r["exact_match"])
    total_prompts = len(results)

    exact_match_ratio = total_exact_matches / total_prompts if total_prompts > 0 else 0

    return {
        "exact_match_ratio": exact_match_ratio,
        "total_exact_matches": total_exact_matches,
        "total_prompts": total_prompts,
    }
