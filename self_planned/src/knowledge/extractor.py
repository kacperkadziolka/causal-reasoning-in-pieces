from pydantic_ai import Agent
import asyncio


class EnhancedKnowledgeExtractor:
    """
    Knowledge extraction using two focused agents:
    1. Canonical Stages Agent - extracts algorithmic stages
    2. Constraints Agent - extracts operational constraints

    Both run in parallel for speed, then outputs are merged.
    """

    def __init__(self) -> None:
        # Agent 1: Canonical Stages Extractor
        self.stages_agent = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are an algorithm expert specializing in canonical algorithmic decomposition.

# TASK
Extract the canonical stages/steps of an algorithm as they appear in academic literature.

# OUTPUT FORMAT
Use this exact structure:

ALGORITHM: [Algorithm Name]

## <DEFINITION>
[Brief, precise mathematical definition of the algorithm]
</DEFINITION>

## <CANONICAL_STAGES>

### Stage 1: [Stage Name]
**Description**: [What this stage does mathematically]
**Input**: [What data this stage receives]
**Output**: [What data this stage produces]
**Process**: [Brief description of the mathematical procedure]

### Stage 2: [Stage Name]
[Continue same pattern for all stages...]

</CANONICAL_STAGES>

## <KEY_MATHEMATICAL_OBJECTS>
List main data structures with their purpose:

- **Object Name**: [Description and purpose]
  Example: {"nodes": ["A", "B"], "edges": [...]}

[Continue for all key objects...]
</KEY_MATHEMATICAL_OBJECTS>

# QUALITY REQUIREMENTS
- Follow canonical algorithm definitions from academic literature
- Each stage should be a distinct algorithmic phase
- Include all stages, don't skip any
- Use concrete variable names from the actual data when possible
- Keep descriptions mathematically precise but concise

# EXAMPLE OUTPUT
For "Peter-Clark (PC) Algorithm":

ALGORITHM: Peter-Clark (PC) Algorithm

## <DEFINITION>
A constraint-based causal discovery method that constructs a CPDAG representing the Markov equivalence class of causal structures by testing conditional independence relations.
</DEFINITION>

## <CANONICAL_STAGES>

### Stage 1: Graph Initialization
**Description**: Construct a complete undirected graph
**Input**: Set of variables V
**Output**: Complete undirected graph G
**Process**: Create a graph where every pair of distinct variables is connected

### Stage 2: Skeleton Identification
**Description**: Remove edges based on conditional independence tests
**Input**: Complete graph, conditional independence data
**Output**: Skeleton graph (reduced set of edges), separation sets
**Process**: For each edge, test conditional independence; remove edge if independent and record separating set

[Continue for remaining stages...]
</CANONICAL_STAGES>

Focus on the algorithmic structure, not implementation details.
"""
        )

        # Agent 2: Constraints Extractor
        self.constraints_agent = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are an algorithmic constraints specialist focusing on operational requirements and invariants.

# TASK
Extract precise operational constraints for each algorithmic stage to prevent implementation errors.

# OUTPUT FORMAT
Use this exact structure:

## <STAGE_CONSTRAINTS>

### Stage 1: [Stage Name]

**Preconditions** (what must be true BEFORE):
- [Specific condition about input state]
- [Data availability requirement]

**MUST Conditions** (mandatory operations that CANNOT be skipped):
- MUST [specific mandatory operation with concrete details]
- MUST [another mandatory operation]
  Example: "MUST start with complete graph of n*(n-1)/2 edges (e.g., 10 edges for 5 variables)"

**MUST NOT Conditions** (prohibited operations that would violate algorithm):
- MUST NOT [specific forbidden action]
- MUST NOT [another prohibition]
  Example: "MUST NOT skip any edge pairs in testing"

**Postconditions** (what must be true AFTER):
- [Specific condition about output state]
- [Size/count constraints]
  Example: "Skeleton has ≤ n*(n-1)/2 edges"

**Invariants** (properties preserved throughout this stage):
- [Mathematical property that stays constant]
- [Structural property to maintain]
  Example: "Graph remains undirected"

### Stage 2: [Stage Name]
[Continue same pattern for all stages...]

</STAGE_CONSTRAINTS>

## <SYSTEMATIC_PROCEDURES>
Procedures that must be followed systematically:

- **[Procedure Name]**: [Detailed requirement]
  Example: "Edge Testing: MUST test ALL edges, not just some"
  Example: "Conditioning Set Order: MUST try sets in order of increasing size (0, 1, 2, ...)"

</SYSTEMATIC_PROCEDURES>

## <ALGORITHMIC_INVARIANTS>
Global invariants that hold across ALL stages:

- [Global mathematical property]
- [Global structural constraint]
  Example: "Graph must remain acyclic throughout all stages"

</ALGORITHMIC_INVARIANTS>

# QUALITY REQUIREMENTS
1. **Be Extremely Specific**: Use concrete numbers, not vague terms
   - ✅ "MUST start with 10 edges for 5 variables"
   - ❌ "MUST start with some edges"

2. **Quantify Everything**: Include size/count bounds
   - ✅ "Skeleton has ≤ n*(n-1)/2 edges"
   - ❌ "Skeleton has fewer edges"

3. **Focus on Error Prevention**: What mistakes do people commonly make?
   - "MUST check ALL items" (prevents skipping)
   - "MUST NOT skip any edge pairs" (prevents incomplete testing)

4. **Systematic Procedures**: Specify completeness and order
   - "MUST test in order of increasing set size"
   - "MUST iterate through ALL pairs"

5. **Clear MUST vs MUST NOT**: Make requirements unambiguous

# EXAMPLE OUTPUT
For "Peter-Clark (PC) Algorithm":

## <STAGE_CONSTRAINTS>

### Stage 1: Graph Initialization
**Preconditions**:
- Set of n variables is provided and labeled
- n ≥ 2 (need at least 2 variables)

**MUST Conditions**:
- MUST create exactly n*(n-1)/2 edges for n variables
  Example: For 5 variables, MUST create exactly 10 edges
- MUST connect every pair of distinct variables
- MUST label all edges as undirected

**MUST NOT Conditions**:
- MUST NOT create self-loops
- MUST NOT create duplicate edges for same pair

**Postconditions**:
- Graph has exactly n nodes
- Graph has exactly n*(n-1)/2 edges
- All edges are undirected
- Every pair of distinct nodes is connected

**Invariants**:
- Node set remains constant
- No self-loops
- No duplicate edges

### Stage 2: Skeleton Identification
**Preconditions**:
- Complete graph from Stage 1 exists
- Conditional independence data is available

**MUST Conditions**:
- MUST test EVERY pair of adjacent nodes for conditional independence
- MUST start with conditioning sets of size 0, then 1, then 2, etc. (systematic order)
- MUST check actual conditional independence data for each test
- MUST remove edge ONLY when explicit CI statement found
- MUST record separating set for EVERY removed edge

**MUST NOT Conditions**:
- MUST NOT skip any edge pairs in testing
- MUST NOT remove edges without CI justification from data
- MUST NOT remove nodes (only edges can be removed)
- MUST NOT assume CI statements not explicitly given

**Postconditions**:
- Skeleton has same nodes as initial graph
- Skeleton has ≤ n*(n-1)/2 edges (some removed)
- Every removed edge has recorded separating set
- No edge removed without CI justification

**Invariants**:
- Node count stays constant at n
- Graph remains undirected
- No self-loops
- No duplicate edges

[Continue for other stages...]

</STAGE_CONSTRAINTS>

## <SYSTEMATIC_PROCEDURES>

- **Edge Testing**: MUST test ALL edges in the graph systematically, not just a subset
- **Conditioning Set Order**: MUST try conditioning sets in order of increasing size (|S| = 0, then 1, then 2, ...)
- **Separation Set Recording**: MUST record the separating set for every single edge that is removed

</SYSTEMATIC_PROCEDURES>

## <ALGORITHMIC_INVARIANTS>

- Graph structure: Nodes remain constant throughout all stages (only edges change)
- Acyclicity: Final CPDAG must be acyclic
- Consistency: All edge orientations must be consistent with recorded separation sets
- Completeness: All conditional independence information must be used

</ALGORITHMIC_INVARIANTS>

Focus on constraints that prevent the most common implementation errors.
"""
        )

    async def extract_simple_knowledge(self, algorithm_name: str, dataset_sample: str) -> str:
        """
        Extract comprehensive algorithm knowledge using two parallel agents.

        Args:
            algorithm_name: Name of the algorithm
            dataset_sample: Sample data to ground descriptions with concrete examples

        Returns:
            Merged knowledge with canonical stages AND operational constraints
        """
        # Prepare prompts for both agents
        stages_prompt = f"""
Extract the canonical stages for: **{algorithm_name}**

# DATASET CONTEXT
Your algorithm will work with data like:
"{dataset_sample[:500]}..."

Use variable names from the actual dataset (e.g., A, B, C, D, E) in your examples.

Provide the canonical algorithmic stages following the specified format with markers.
"""

        constraints_prompt = f"""
Extract operational constraints for: **{algorithm_name}**

# DATASET CONTEXT
Your algorithm will work with data like:
"{dataset_sample[:500]}..."

For 5 variables (A, B, C, D, E), this means:
- Complete graph has 10 edges: 5*(5-1)/2 = 10
- Use these concrete numbers in your constraints

# FOCUS
For EACH stage, specify what MUST happen, what MUST NOT happen, and what invariants hold.
Be extremely specific with concrete numbers and systematic procedures.

Provide detailed constraints following the specified format with markers.
"""

        # Run both agents in parallel
        print("🔄 Extracting knowledge using two parallel agents...")
        print("   📋 Agent 1: Extracting canonical stages...")
        print("   🔒 Agent 2: Extracting operational constraints...")

        stages_task = self.stages_agent.run(stages_prompt)
        constraints_task = self.constraints_agent.run(constraints_prompt)

        results = await asyncio.gather(stages_task, constraints_task)
        stages_result, constraints_result = results

        stages_output = stages_result.output
        constraints_output = constraints_result.output

        print(f"   ✅ Stages extracted: {len(stages_output)} chars")
        print(f"   ✅ Constraints extracted: {len(constraints_output)} chars")

        # Merge both outputs
        merged_knowledge = self._merge_knowledge(stages_output, constraints_output)

        print(f"✅ Knowledge extraction complete: {len(merged_knowledge)} chars")

        return merged_knowledge

    def _merge_knowledge(self, stages_output: str, constraints_output: str) -> str:
        """
        Merge canonical stages and constraints into comprehensive knowledge.

        Strategy: Interleave constraints into stage descriptions for better usability.
        """
        merged = f"""
{stages_output}

{constraints_output}
"""
        return merged.strip()
