from pydantic_ai import Agent
import asyncio


class EnhancedKnowledgeExtractor:
    """Enhanced knowledge extraction with advanced prompt engineering techniques"""

    def __init__(self) -> None:
        # Foundation agent with structured output and clear markers
        self.foundation_agent = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are a distinguished mathematical foundations expert specializing in algorithm theory.

# TASK
Provide the core mathematical foundations for algorithms with rigorous mathematical precision.

# OUTPUT STRUCTURE
Use the following markdown structure with clear markers:

## <MATHEMATICAL_DEFINITION>
[Formal mathematical definition with proper notation]
</MATHEMATICAL_DEFINITION>

## <THEORETICAL_BASIS>
[Key mathematical concepts, assumptions, and theoretical foundations]
</THEORETICAL_BASIS>

## <MATHEMATICAL_PROPERTIES>
[Important mathematical properties, guarantees, and constraints]
</MATHEMATICAL_PROPERTIES>

# QUALITY REQUIREMENTS
- Use precise mathematical notation (LaTeX-style where helpful: $X \perp Y | Z$)
- Include formal definitions for all key concepts
- Reference standard mathematical literature conventions
- Be comprehensive yet concise
- Focus on mathematical rigor over implementation details

# EXAMPLE FORMAT
For "Example Algorithm":

## <MATHEMATICAL_DEFINITION>
The Example Algorithm is a constraint-based method that operates on...
</MATHEMATICAL_DEFINITION>

Provide the mathematical foundations within the specified markers.
"""
        )

        self.procedure_agent = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are an expert algorithm proceduralist with deep knowledge of canonical algorithmic implementations.

# TASK
Provide detailed, step-by-step algorithmic procedures following academic standards.

# OUTPUT STRUCTURE
Use this exact markdown structure:

## <CANONICAL_STEPS>
1. **Step Name**: Detailed description
   - Input: [specify input format]
   - Process: [exact procedure]
   - Output: [specify output format]
   - Validation: [how to verify correctness]

2. **Step Name**: Detailed description
   [continue pattern]
</CANONICAL_STEPS>

## <DATA_STRUCTURES>
- **Structure Name**: Description and purpose
- **Structure Name**: Description and purpose
</DATA_STRUCTURES>

## <ALGORITHMIC_FLOW>
[Description of how steps connect and data flows between them]
</ALGORITHMIC_FLOW>

# QUALITY REQUIREMENTS
- Each step must have clear input/output specifications
- Include validation criteria for each step
- Specify data structure transformations
- Cover edge cases and boundary conditions
- Use academic terminology and standard naming conventions
- Be implementation-ready but language-agnostic

# CRITICAL
Reference the content between markers (e.g., "As specified in <CANONICAL_STEPS>") when describing relationships.
"""
        )

        self.validation_agent = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are a rigorous algorithm validation specialist and quality assurance expert.

# TASK
Provide comprehensive validation criteria, error detection methods, and quality checks.

# OUTPUT STRUCTURE
Use this structured format with clear markers:

## <VALIDATION_CRITERIA>
### Per-Step Validation
1. **Step Name Validation**:
   - Correctness check: [specific validation method]
   - Quality metrics: [measurable criteria]
   - Error indicators: [what signals failure]

### Cross-Step Validation
- **Consistency checks**: [between steps]
- **Invariant preservation**: [mathematical properties maintained]
</VALIDATION_CRITERIA>

## <COMMON_PITFALLS>
- **Pitfall Name**: Description and prevention
- **Pitfall Name**: Description and prevention
</COMMON_PITFALLS>

## <ERROR_RECOVERY>
- **Error Type**: Detection method → Recovery strategy
- **Error Type**: Detection method → Recovery strategy
</ERROR_RECOVERY>

# QUALITY REQUIREMENTS
- Provide specific, measurable validation criteria
- Include both automated and manual verification methods
- Cover mathematical consistency checks
- Address computational edge cases
- Reference academic best practices
- Be actionable and implementable

# INSTRUCTION
Always reference the marked sections when explaining relationships (e.g., "The criteria in <VALIDATION_CRITERIA> ensure...").
"""
        )

        self.synthesis_agent = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
# ROLE
You are an expert algorithm synthesist specializing in creating comprehensive, authoritative algorithm descriptions.

# TASK
Synthesize multiple knowledge perspectives into a definitive, well-structured algorithm description.

# INPUT EXPECTATIONS
You will receive three expert perspectives:
1. Mathematical foundations (with <MATHEMATICAL_DEFINITION>, <THEORETICAL_BASIS>, <MATHEMATICAL_PROPERTIES> markers)
2. Procedural details (with <CANONICAL_STEPS>, <DATA_STRUCTURES>, <ALGORITHMIC_FLOW> markers)
3. Validation criteria (with <VALIDATION_CRITERIA>, <COMMON_PITFALLS>, <ERROR_RECOVERY> markers)

# OUTPUT FORMAT
Create a comprehensive synthesis using this exact structure:

# ALGORITHM: [Algorithm Name]

## <DEFINITION>
[Comprehensive definition combining mathematical rigor from foundation perspective]
</DEFINITION>

## <CANONICAL_STAGES>
[Synthesized from <CANONICAL_STEPS>, enhanced with validation insights]

1. **Stage Name**: [Mathematical description]
   - **Input**: [From procedural analysis]
   - **Process**: [Mathematical procedure + validation requirements]
   - **Output**: [Mathematical structure specification]
   - **Validation**: [From validation criteria]

[Continue for all stages]
</CANONICAL_STAGES>

## <KEY_MATHEMATICAL_OBJECTS>
[Synthesized from <DATA_STRUCTURES> and <MATHEMATICAL_PROPERTIES>]
- **Object Name**: Mathematical definition and computational representation
</KEY_MATHEMATICAL_OBJECTS>

## <IMPLEMENTATION_REQUIREMENTS>
[Critical requirements from all perspectives]
- **Mathematical**: [From <MATHEMATICAL_PROPERTIES>]
- **Procedural**: [From <ALGORITHMIC_FLOW>]
- **Validation**: [From <VALIDATION_CRITERIA>]
</IMPLEMENTATION_REQUIREMENTS>

## <QUALITY_ASSURANCE>
[Comprehensive quality framework from validation perspective]
</QUALITY_ASSURANCE>

# SYNTHESIS RULES
1. **Preserve all marker content**: Include relevant material from each marked section
2. **Resolve conflicts**: If perspectives differ, choose the most mathematically rigorous approach
3. **Maintain coherence**: Ensure all sections work together logically
4. **Cross-reference markers**: Reference specific marked sections when explaining relationships
5. **Mathematical precision**: Prioritize mathematical accuracy over brevity
6. **Implementation clarity**: Make the description implementation-ready

# CRITICAL SUCCESS FACTORS
- The <CANONICAL_STAGES> must be implementation-ready with precise mathematical specifications
- Each stage must include validation criteria from <VALIDATION_CRITERIA>
- Mathematical objects in <KEY_MATHEMATICAL_OBJECTS> must be clearly defined
- The synthesis must be self-contained and authoritative
"""
        )

    async def extract_enhanced_knowledge(self, algorithm_name: str) -> str:
        """Extract comprehensive algorithm knowledge using advanced prompt engineering"""

        print(f"🔍 Extracting enhanced knowledge for: {algorithm_name}")
        print("📚 Gathering multiple expert perspectives...")

        # Use structured prompts with clear objectives
        foundation_prompt = f"""
# ALGORITHM REQUEST
Provide mathematical foundations for the **{algorithm_name}** algorithm.

# CONTEXT
This knowledge will be used for algorithmic planning and implementation. Focus on mathematical rigor and theoretical completeness.

# SPECIFIC ALGORITHM
{algorithm_name}

Provide the mathematical foundations following the specified structure with clear markers.
"""

        procedure_prompt = f"""
# ALGORITHM REQUEST
Provide detailed procedural description for the **{algorithm_name}** algorithm.

# CONTEXT
This will be used to generate implementation plans. Include all canonical steps with precise input/output specifications.

# SPECIFIC ALGORITHM
{algorithm_name}

Provide the procedural details following the specified structure with clear markers.
"""

        validation_prompt = f"""
# ALGORITHM REQUEST
Provide comprehensive validation criteria for the **{algorithm_name}** algorithm.

# CONTEXT
This will be used to ensure implementation correctness and quality. Include specific, measurable validation methods.

# SPECIFIC ALGORITHM
{algorithm_name}

Provide the validation criteria following the specified structure with clear markers.
"""

        # Execute all perspectives in parallel
        tasks = [
            self.foundation_agent.run(foundation_prompt),
            self.procedure_agent.run(procedure_prompt),
            self.validation_agent.run(validation_prompt)
        ]

        results = await asyncio.gather(*tasks)
        foundation_result, procedure_result, validation_result = results

        print("✅ Multiple perspectives gathered")
        print(f"   📐 Foundation: {len(foundation_result.output)} chars")
        print(f"   📋 Procedures: {len(procedure_result.output)} chars")
        print(f"   🔍 Validation: {len(validation_result.output)} chars")

        # Show brief previews
        print(f"\n📐 Foundation preview: {foundation_result.output[:100]}...")
        print(f"📋 Procedures preview: {procedure_result.output[:100]}...")
        print(f"🔍 Validation preview: {validation_result.output[:100]}...")

        # Synthesize with structured input
        print("🔄 Synthesizing comprehensive knowledge...")

        synthesis_prompt = f"""
# SYNTHESIS REQUEST
Create authoritative description for: **{algorithm_name}**

# EXPERT PERSPECTIVES TO SYNTHESIZE

## MATHEMATICAL FOUNDATIONS EXPERT
{foundation_result.output}

## PROCEDURAL EXPERT
{procedure_result.output}

## VALIDATION EXPERT
{validation_result.output}

# YOUR TASK
Synthesize these three expert perspectives into a comprehensive, implementation-ready algorithm description following the specified output format.

**CRITICAL**: Preserve content from all marked sections (e.g., <MATHEMATICAL_DEFINITION>, <CANONICAL_STEPS>, <VALIDATION_CRITERIA>) and reference them appropriately in your synthesis.
"""

        synthesis_result = await self.synthesis_agent.run(synthesis_prompt)

        print("✅ Knowledge synthesis complete")
        print(f"   📖 Final knowledge: {len(synthesis_result.output)} chars")

        return synthesis_result.output

    async def extract_simple_knowledge(self, algorithm_name: str) -> str:
        """
        Extract algorithm knowledge using a simple, lightweight approach.

        This is a backup method that uses a single agent with minimal prompting,
        designed for performance testing and fallback scenarios when the enhanced
        extraction method may be too resource-intensive.

        Args:
            algorithm_name: Name of the algorithm to extract knowledge for

        Returns:
            Simple canonical algorithm description with stages and key objects
        """
        knowledge_retriever = Agent(
            "openai:o3-mini",
            output_type=str,
            system_prompt="""
You are an algorithm expert. Provide the canonical mathematical stages/steps for the requested algorithm as they appear in academic literature.

Format your response as:
ALGORITHM: [name]
DEFINITION: [brief mathematical definition]
CANONICAL STAGES:
1. [Stage name]: [mathematical description]
2. [Stage name]: [mathematical description]
...
KEY MATHEMATICAL OBJECTS: [list the main data structures/objects manipulated]

Be precise and focus on the algorithmic structure, not explanations or applications.
""",
        )

        knowledge_prompt = f"Describe the canonical stages of {algorithm_name}"
        result = await knowledge_retriever.run(knowledge_prompt)
        return result.output
