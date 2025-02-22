import re

from ToT.utils import config


def extract_edges(answer: str) -> set[tuple]:
    answer = answer.replace("\r\n", "\n").replace("\r", "\n")

    markers = config.get("markers")
    if not markers:
        raise ValueError("No markers provided in config.")

    # Escape markers for regex and join them with OR.
    escaped_markers = [re.escape(marker) for marker in markers]
    markers_pattern = r"|".join(escaped_markers)

    # Build a regex that looks for:
    #   "Step 4:" or "Step 5:" followed by any text (non-greedy)
    pattern = r"Step\s+(4|5)(?::|(?:\s*-))\s*.*?(?:" + markers_pattern + r")\s*\n"

    marker_match = re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
    if not marker_match:
        raise ValueError(
            "No valid marker with 'Step 4:' or 'Step 5:' and one of the config markers found in the answer. Failed to extract edges.")
    start_index = marker_match.end()

    lines = answer[start_index:].splitlines()
    adjacency_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("- Node"):
            adjacency_lines.append(stripped_line)
        elif adjacency_lines:
            break

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

    for line in adjacency_lines:
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
