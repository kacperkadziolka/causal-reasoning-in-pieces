import re


def extract_edges_incident_format(answer: str) -> set:
    """
    Extract the causal edges from the provided LLM answer string using the adjacency list format.

    :param answer: The answer returned by the LLM API.
    :return: A set of edges extracted from the answer, represented as sorted tuples.
    """
    try:
        # Normalize line breaks
        answer = answer.replace("\r\n", "\n").replace("\r", "\n")

        # Locate Step 5
        step_5_pattern = r"Step 5: Compile the Causal Undirected Skeleton"
        step_5_match = re.search(step_5_pattern, answer, flags=re.IGNORECASE)
        if not step_5_match:
            raise ValueError("Step 5 section not found in the answer.")

        # Slice the answer from right after Step 5
        adjacency_section = answer[step_5_match.end():].splitlines()

        """
        Below code works well with the few-shot example, where the answer are more consistent with
        requested format. In zer-shot settings, model often fails to includes the "In this graph:" 
        line in the output.
        """
        # # Find the start of the adjacency list
        # adjacency_start = answer.find("In this graph:", step_5_match.end())
        # if adjacency_start == -1:
        #     raise ValueError("Adjacency list section not found in Step 5.")
        #
        # # Split lines from the adjacency list
        # adjacency_section = answer[adjacency_start:].splitlines()

        edges = set()

        """
        Following code works well with the few-shot example, where the answer are more consistent,
        and each starts with - Node. In zero-shot settings, model often fails to include the "node"
        word.
        """
        # Pattern for lines:
        #   - Node X has no connections.
        no_conn_pattern = re.compile(r"^- Node\s+(\w+)\s+has\s+no\s+connections\.$", re.IGNORECASE)

        # Pattern to match lines:
        #   - Node A is connected to node C.
        #   - Node A is connected only to node D.
        #   - Node B is connected to nodes A, C and D.
        #   - Node C is connected to node A and node B.
        conn_pattern = re.compile(
            r"^- Node\s+(\w+)\s+is\s+connected(?:\s+only)?\s+to\s+nodes?\s+(.+)\.$",
            re.IGNORECASE
        )

        # no_conn_pattern = re.compile(
        #     r"^-\s*(?:Node\s+)?(\w+)\s+has\s+no\s+connections\.?",
        #     re.IGNORECASE
        # )
        # conn_pattern = re.compile(
        #     r"^-\s*(?:Node\s+)?(\w+)\s+is\s+connected\s+to\s+(?:node[s]?\s+)?(.+?)\.?\s*$",
        #     re.IGNORECASE
        # )

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
