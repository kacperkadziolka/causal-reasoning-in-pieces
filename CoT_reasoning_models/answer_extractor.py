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


def extract_hypothesis(text: str) -> str:
    """
    Extracts Hypothesis from the input text.

    Parameters:
    - text (str): The input text containing "Hypothesis:".

    Returns:
    - str: The extracted hypothesis.
    """
    # Use regex to extract everything after "Hypothesis:"
    match = re.search(r"Hypothesis:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If "Hypothesis:" is not found, return empty string
        return ""


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


def extract_separation_sets(premise: str):
    """Extract separation sets from premise text."""
    sep_sets = {}

    # Process conditional independence statements first
    cond_pattern = r"([A-E]) and ([A-E]) are independent given ([^\.]+)"
    cond_matches = re.findall(cond_pattern, premise)

    for X, Y, condition_text in cond_matches:
        # Extract individual variables from the conditioning set
        cond_vars = re.findall(r'[A-E]', condition_text)
        sep_sets[frozenset([X, Y])] = set(cond_vars)

    # Process marginal independence statements
    marg_patterns = [
        r"([A-E]) is independent of ([A-E])",
        r"However,\s+([A-E]) is independent of ([A-E])"  # Handle "However, " prefix
    ]

    for pattern in marg_patterns:
        marg_matches = re.findall(pattern, premise)
        for X, Y in marg_matches:
            key = frozenset([X, Y])
            # Only add if not already set by a conditional independence
            if key not in sep_sets:
                sep_sets[key] = set()

    return sep_sets


def find_v_structures(skeleton_edges, separation_sets):
    """
    Find V-structures in a causal graph based on separation sets.

    A v-structure occurs when two non-adjacent nodes X and Y both point to a third node Z,
    and Z is NOT in the separation set of X and Y.
    """
    # Convert to adjacency list
    adjacency = {}
    for edge in skeleton_edges:
        u, v = edge
        if u not in adjacency:
            adjacency[u] = set()
        if v not in adjacency:
            adjacency[v] = set()
        adjacency[u].add(v)
        adjacency[v].add(u)

    v_structures = []
    nodes = set(node for edge in skeleton_edges for node in edge)

    for z in nodes:
        neighbors = list(adjacency.get(z, []))

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                x, y = neighbors[i], neighbors[j]

                # Check if X and Y are not adjacent
                if y not in adjacency.get(x, set()):
                    pair = frozenset([x, y])
                    if pair in separation_sets:
                        sep_set = separation_sets[pair]

                        # V-structure occurs when Z is NOT in the separation set
                        # The previous condition "and not sep_set" was incorrect
                        # as it required the separation set to be empty
                        if z not in sep_set:
                            v_structures.append((x, z, y))

    return v_structures


def extract_vstructures_json(answer: str) -> list:
    """
    Extract the v-structures from the provided LLM answer string using the JSON format.

    :param answer: The answer returned by the LLM API.
    :return: A list of v-structures extracted from the answer.
    """
    try:
        # First approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)

            # Extract v-structures
            if "v_structures" in data:
                v_structures = [tuple(v_struct) for v_struct in data["v_structures"]]
                return v_structures

        # Second approach: Find all potential JSON objects and test each one
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "v_structures" in data:
                    v_structures = [tuple(v_struct) for v_struct in data["v_structures"]]
                    return v_structures
            except json.JSONDecodeError:
                continue

        # If no v_structures found in JSON, look for v-structures in text format
        v_structures_pattern = r'v.?structures?:?\s*\[(.*?)\]'
        v_struct_match = re.search(v_structures_pattern, answer, re.IGNORECASE | re.DOTALL)
        if v_struct_match:
            content = v_struct_match.group(1)
            # Extract arrays like ["A", "B", "C"]
            array_pattern = r'\[\s*"([A-E])"\s*,\s*"([A-E])"\s*,\s*"([A-E])"\s*\]'
            v_structures = []
            for match in re.finditer(array_pattern, content):
                v_structures.append(tuple(match.groups()))
            if v_structures:
                return v_structures

        raise ValueError("No v-structures found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract v-structures: {e}")


def compare_vstructures(expected_vstructs: list, answer_vstructs: list) -> dict:
    """
    Compare the expected v-structures with the model's predicted v-structures.
    A v-structure (X,Z,Y) is equivalent to (Y,Z,X) since both represent X->Z<-Y.

    :param expected_vstructs: The ground-truth v-structures.
    :param answer_vstructs: The model's predicted v-structures.
    :return: A dictionary with metrics for comparison.
    """

    # Normalize v-structures by ensuring the middle node is fixed and the endpoints are sorted
    def normalize_v_structure(v_struct):
        if len(v_struct) != 3:
            return v_struct
        # Keep the middle node (collider) fixed but sort the endpoints
        endpoints = sorted([v_struct[0], v_struct[2]])
        return (endpoints[0], v_struct[1], endpoints[1])

    # Normalize and convert to set for comparison
    expected_set = {normalize_v_structure(v) for v in expected_vstructs}
    answer_set = {normalize_v_structure(v) for v in answer_vstructs}

    # Check if the v-structures are an exact match
    exact_match = expected_set == answer_set

    if exact_match:
        print("The model's predicted v-structures match the expected v-structures!")
    else:
        print("Mismatch found:")
        print(f"Expected v-structures: {expected_set}")
        print(f"Model v-structures: {answer_set}")
        print(f"Missing in model: {expected_set - answer_set}")
        print(f"Extra in model: {answer_set - expected_set}")

    # Calculate detailed metrics
    true_positive = len(expected_set & answer_set)
    false_positive = len(answer_set - expected_set)
    false_negative = len(expected_set - answer_set)

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "expected_count": len(expected_set),
        "predicted_count": len(answer_set),
        "exact_match": exact_match,
        "missing_vstructs": list(expected_set - answer_set),
        "extra_vstructs": list(answer_set - expected_set)
    }

def extract_directed_edges_json(answer: str) -> list:
    """
    Extract the directed edges from the provided LLM answer string using the JSON format.

    :param answer: The answer returned by the LLM API.
    :return: A list of directed edges (tuples representing source->target) extracted from the answer.
    """
    try:
        # First approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)

            # Extract directed edges
            if "directed_edges" in data:
                directed_edges = [tuple(edge) for edge in data["directed_edges"]]
                return directed_edges

        # Second approach: Find all potential JSON objects and test each one
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "directed_edges" in data:
                    directed_edges = [tuple(edge) for edge in data["directed_edges"]]
                    return directed_edges
            except json.JSONDecodeError:
                continue

        # If no directed_edges found in JSON, look for directed edges in text format
        edges_pattern = r'directed.?edges?:?\s*\[(.*?)\]'
        edges_match = re.search(edges_pattern, answer, re.IGNORECASE | re.DOTALL)
        if edges_match:
            content = edges_match.group(1)
            # Extract arrays like ["A", "B"] representing A->B
            array_pattern = r'\[\s*"([A-E])"\s*,\s*"([A-E])"\s*\]'
            directed_edges = []
            for match in re.finditer(array_pattern, content):
                directed_edges.append(tuple(match.groups()))
            if directed_edges:
                return directed_edges

        raise ValueError("No directed edges found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract directed edges: {e}")


def extract_directed_edges_literal_format_json(answer: str) -> list:
    """
    Extract the directed edges from the provided LLM answer string using the expected JSON format.

    Expected JSON format:
    {
      "final_graph": {
        "directed_edges": [
          {
            "from": "Node1",
            "to": "Node2"
          },
          {
            "from": "Node2",
            "to": "Node3"
          }
        ],
        "undirected_edges": [
          ["Node3", "Node4"],
          ["Node5", "Node6"]
        ]
      },
      "orientation_steps": [
        {
          "step": 1,
          "rule_applied": "R1",
          "edge_oriented": {
            "from": "Node2",
            "to": "Node3"
          },
          "explanation": "Applied Rule R1 because there exists an edge Node1 → Node2 and an undirected edge Node2 – Node3, with Node1 not adjacent to Node3."
        },
        {
          "step": 2,
          "rule_applied": "R2",
          "edge_oriented": {
            "from": "Node4",
            "to": "Node5"
          },
          "explanation": "Applied Rule R2 due to the chain Node4 → Node6 → Node5, with no direct edge between Node4 and Node5."
        }
      ]
    }

    :param answer: The answer returned by the LLM API.
    :return: A list of directed edges (tuples representing source->target) extracted from the answer.
    """
    try:
        # First approach: Find JSON block delimited by triple backticks.
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            if "final_graph" in data and "directed_edges" in data["final_graph"]:
                directed_edges = []
                for edge in data["final_graph"]["directed_edges"]:
                    if "from" in edge and "to" in edge:
                        directed_edges.append((edge["from"], edge["to"]))
                if directed_edges:
                    return directed_edges

        # Second approach: Search for any JSON objects within the answer.
        json_blocks = re.findall(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', answer)
        for json_str in json_blocks:
            try:
                data = json.loads(json_str)
                if "final_graph" in data and "directed_edges" in data["final_graph"]:
                    directed_edges = []
                    for edge in data["final_graph"]["directed_edges"]:
                        if "from" in edge and "to" in edge:
                            directed_edges.append((edge["from"], edge["to"]))
                    if directed_edges:
                        return directed_edges
            except json.JSONDecodeError:
                continue

        # Third approach: Look for directed edges in text format from the final_graph section.
        edges_pattern = r'"directed_edges"\s*:\s*\[(.*?)\]'
        edges_match = re.search(edges_pattern, answer, re.IGNORECASE | re.DOTALL)
        if edges_match:
            content = edges_match.group(1)
            # Each edge is expected to be in the format {"from": "X", "to": "Y"}
            edge_pattern = r'\{\s*"from"\s*:\s*"([^"]+)"\s*,\s*"to"\s*:\s*"([^"]+)"\s*\}'
            directed_edges = []
            for match in re.finditer(edge_pattern, content):
                directed_edges.append((match.group(1), match.group(2)))
            if directed_edges:
                return directed_edges

        raise ValueError("No directed edges found in the answer.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract directed edges: {e}")


def compare_directed_edges(expected_edges: list, model_edges: list) -> dict:
    """
    Compare the expected directed edges with the model's predicted directed edges.
    A directed edge (X,Y) represents X→Y and is different from (Y,X).

    :param expected_edges: The ground-truth directed edges.
    :param model_edges: The model's predicted directed edges.
    :return: A dictionary with metrics for comparison.
    """
    # Convert to sets for comparison
    expected_set = set(expected_edges)
    model_set = set(model_edges)

    # Check if the directed edges are an exact match
    exact_match = expected_set == model_set

    # Find reversed edges (when direction is flipped)
    reversed_edges = []
    for edge in expected_set:
        reversed = (edge[1], edge[0])
        if reversed in model_set and edge not in model_set:
            reversed_edges.append((edge, reversed))

    if exact_match:
        print("The model's predicted directed edges match the expected directed edges!")
    else:
        print("Mismatch found:")
        print(f"Expected directed edges: {expected_set}")
        print(f"Model directed edges: {model_set}")
        print(f"Missing in model: {expected_set - model_set}")
        print(f"Extra in model: {model_set - expected_set}")
        if reversed_edges:
            print(f"Reversed edges (expected→actual): {reversed_edges}")

    # Calculate detailed metrics
    true_positive = len(expected_set & model_set)
    false_positive = len(model_set - expected_set)
    false_negative = len(expected_set - model_set)

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "expected_count": len(expected_set),
        "predicted_count": len(model_set),
        "exact_match": exact_match,
        "missing_edges": list(expected_set - model_set),
        "extra_edges": list(model_set - expected_set),
        "reversed_edges": reversed_edges
    }


def extract_hypothesis_answer(answer: str) -> bool:
    """
    Extract the hypothesis answer (True/False) from the provided LLM answer string.

    Parameters:
    - answer (str): The answer returned by the LLM API.

    Returns:
    - bool: The extracted hypothesis answer as a Python boolean.

    Raises:
    - RuntimeError: If extraction fails.
    """
    try:
        # First approach: Try to parse the entire answer as JSON
        try:
            data = json.loads(answer)
            if "hypothesis_answer" in data:
                return bool(data["hypothesis_answer"])
        except json.JSONDecodeError:
            pass

        # Second approach: Find JSON block with triple quotes
        json_match = re.search(r'```(?:json)?\s*(.*?)```', answer, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                if "hypothesis_answer" in data:
                    return bool(data["hypothesis_answer"])
            except json.JSONDecodeError:
                pass

        # Third approach: Try to find complete JSON objects in the text
        potential_json_start = 0
        while potential_json_start < len(answer):
            start_idx = answer.find('{', potential_json_start)
            if start_idx == -1:
                break

            # Find matching closing brace by tracking brace balance
            open_braces = 1
            for end_idx in range(start_idx + 1, len(answer)):
                if answer[end_idx] == '{':
                    open_braces += 1
                elif answer[end_idx] == '}':
                    open_braces -= 1

                if open_braces == 0:
                    # Found a potential JSON object
                    try:
                        json_str = answer[start_idx:end_idx + 1]
                        data = json.loads(json_str)
                        if "hypothesis_answer" in data:
                            return bool(data["hypothesis_answer"])
                    except json.JSONDecodeError:
                        pass
                    break

            potential_json_start = start_idx + 1

        raise ValueError("No valid JSON with hypothesis_answer found")
    except Exception as e:
        raise RuntimeError(f"Failed to extract hypothesis answer: {e}")


def compare_hypothesis_answers(expected_answer: bool, model_answer: bool) -> dict:
    """
    Compare the expected hypothesis answer with the model's predicted answer.

    Parameters:
    - expected_answer (bool): The ground-truth answer (True/False).
    - model_answer (bool): The model's predicted answer (True/False).

    Returns:
    - dict: A dictionary with comparison metrics.
    """
    # Check if the answers match
    exact_match = expected_answer == model_answer

    # Calculate binary classification metrics
    true_positive = 1 if expected_answer and model_answer else 0
    true_negative = 1 if not expected_answer and not model_answer else 0
    false_positive = 1 if not expected_answer and model_answer else 0
    false_negative = 1 if expected_answer and not model_answer else 0

    if exact_match:
        print("\nThe model's predicted answer matches the expected answer!")
    else:
        print("\nMismatch found:")
        print(f"Expected answer: {expected_answer}")
        print(f"Model answer: {model_answer}")

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "exact_match": exact_match
    }


def aggregate_metrics_single_prompt(results: list) -> dict:
    """
    Aggregate metrics from multiple binary classification experiments (True/False predictions).

    This function calculates key classification metrics including accuracy, precision,
    recall, and F1-score. It also tracks class-specific performance and experiment success rates.

    Parameters:
    - results (list): A list of comparison results from multiple experiments.

    Returns:
    - dict: Key metrics organized in categories:
      - summary: Overall performance indicators:
          - accuracy: Percentage of all predictions that were correct (TP+TN)/total
          - f1_score: Harmonic mean of precision and recall
          - exact_match_rate: Percentage of experiments with correct predictions

      - experiment_stats: Basic experiment information:
          - total_samples: Number of hypothesis tests performed
          - exact_matches: Count of correct predictions
          - total_experiments: Number of experiment runs

      - class_metrics: Class-specific performance:
          - precision: When model predicts True, how often it's correct (TP/(TP+FP))
          - recall: What percentage of actual True cases were caught (TP/(TP+FN))
          - specificity: What percentage of actual False cases were correct (TN/(TN+FP))
          - true_class_accuracy: Success rate on "True" hypothesis samples
          - false_class_accuracy: Success rate on "False" hypothesis samples

      - confusion_matrix: Raw classification counts:
          - true_positive: Correctly predicted True hypotheses
          - true_negative: Correctly predicted False hypotheses
          - false_positive: Incorrectly predicted True (Type I error)
          - false_negative: Incorrectly predicted False (Type II error)
          - actual_true/false: Total number of True/False ground truth samples
    """
    if not results:
        return {"error": "No results to aggregate"}

    # Count basic classification outcomes
    tp = sum(r["true_positive"] for r in results)
    tn = sum(r["true_negative"] for r in results)
    fp = sum(r["false_positive"] for r in results)
    fn = sum(r["false_negative"] for r in results)

    # Key derived metrics
    total_samples = tp + tn + fp + fn
    total_actual_true = tp + fn
    total_actual_false = tn + fp

    # Calculate standard performance metrics
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / total_actual_false if total_actual_false > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Success rates by class
    true_class_accuracy = tp / total_actual_true if total_actual_true > 0 else 0
    false_class_accuracy = tn / total_actual_false if total_actual_false > 0 else 0

    # Exact matches
    exact_matches = sum(1 for r in results if r["exact_match"])

    return {
        "summary": {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "exact_match_rate": exact_matches / len(results) if results else 0
        },
        "experiment_stats": {
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "total_experiments": len(results)
        },
        "class_metrics": {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "true_class_accuracy": true_class_accuracy,
            "false_class_accuracy": false_class_accuracy
        },
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "actual_true": total_actual_true,
            "actual_false": total_actual_false
        }
    }
