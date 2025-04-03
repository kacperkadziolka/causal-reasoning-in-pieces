def apply_rule1(pdag):
    """Rule 1: If a→b−c and a,c not adjacent, then orient b→c"""
    changes = False

    for b in pdag:
        # Find directed edges a→b
        parents = [a for a in pdag[b] if pdag[b][a] == '<']
        # Find undirected edges b−c
        undirected = [c for c in pdag[b] if pdag[b][c] == '-']

        for a in parents:
            for c in undirected:
                # Check if a and c are not adjacent
                if c not in pdag[a]:
                    # Orient b→c
                    pdag[b][c] = '>'
                    pdag[c][b] = '<'
                    changes = True

    return changes


def apply_rule2(pdag):
    """Rule 2: If a→b→c and a−c, then orient a→c"""
    changes = False

    for b in pdag:
        # Find directed edges a→b
        parents = [a for a in pdag[b] if pdag[b][a] == '<']
        # Find directed edges b→c
        children = [c for c in pdag[b] if pdag[b][c] == '>']

        for a in parents:
            for c in children:
                # Check if a−c exists
                if c in pdag[a] and pdag[a][c] == '-':
                    # Orient a→c
                    pdag[a][c] = '>'
                    pdag[c][a] = '<'
                    changes = True

    return changes


def apply_rule3(pdag):
    """Rule 3: If a−b, a−c, b→d, c→d, and b,c not adjacent, orient a→d"""
    changes = False

    for a in pdag:
        # Find undirected neighbors of a
        undirected = [n for n in pdag[a] if pdag[a][n] == '-']

        for d in undirected:
            # Find nodes that point to d
            d_parents = [p for p in pdag[d] if pdag[d][p] == '<']

            # Check pairs of d's parents
            for i, b in enumerate(d_parents):
                for c in d_parents[i + 1:]:
                    # Check if b and c are undirected neighbors of a
                    if (b in undirected and c in undirected and
                            # Check if b and c are not adjacent
                            c not in pdag[b]):
                        # Orient a→d
                        pdag[a][d] = '>'
                        pdag[d][a] = '<'
                        changes = True
                        break

    return changes


def apply_rule4(pdag):
    """Rule 4: If a−b, a−c, a−d, b→d, c→d, then orient a→b"""
    changes = False

    for a in pdag:
        # Find undirected neighbors of a
        undirected = [n for n in pdag[a] if pdag[a][n] == '-']

        for b in undirected:
            # Find directed children of b
            b_children = [c for c in pdag[b] if pdag[b][c] == '>']

            for d in b_children:
                if d in undirected:  # a−d exists
                    # Find other undirected neighbors of a that point to d
                    for c in undirected:
                        if (c != b and c != d and
                                d in pdag[c] and pdag[c][d] == '>'):
                            # Orient a→b
                            pdag[a][b] = '>'
                            pdag[b][a] = '<'
                            changes = True
                            break

    return changes


def apply_meek_rules(skeleton_edges, v_structures) -> set[tuple[str, str]]:
    """
    Apply Meek rules to orient edges in a causal graph.

    Args:
        skeleton_edges: Set of tuples representing undirected edges (a, b)
        v_structures: Set/list of tuples (a, b, c) where a → b ← c is a v-structure

    Returns:
        Set of directed edges representing the maximally oriented graph.
    """
    # Build the original adjacency list (if needed by other rules)
    orig_adj = {}
    for u, v in skeleton_edges:
        orig_adj.setdefault(u, set()).add(v)
        orig_adj.setdefault(v, set()).add(u)

    # Convert to PDAG representation
    pdag = {}
    # Initialize with undirected edges
    for u, v in skeleton_edges:
        if u not in pdag:
            pdag[u] = {}
        if v not in pdag:
            pdag[v] = {}
        pdag[u][v] = '-'  # Undirected
        pdag[v][u] = '-'  # Undirected

    # Orient edges according to v-structures
    for x, z, y in v_structures:
        pdag[x][z] = '>'  # x → z
        pdag[z][x] = '<'  # z ← x
        pdag[y][z] = '>'  # y → z
        pdag[z][y] = '<'  # z ← y

    # Apply rules until no more changes
    changes = True
    while changes:
        changes = False

        # Use the modified Rule 1 (without the non-adjacency check)
        changes |= apply_rule1(pdag)
        changes |= apply_rule2(pdag)
        changes |= apply_rule3(pdag)
        changes |= apply_rule4(pdag)

    # Extract directed edges
    directed_edges = set()
    for u in pdag:
        for v in pdag[u]:
            if pdag[u][v] == '>':
                directed_edges.add((u, v))

    return directed_edges
