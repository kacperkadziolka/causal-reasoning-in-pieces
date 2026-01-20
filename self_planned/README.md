# Self-Planned Execution Framework

A generic AI system that automatically detects algorithms in task descriptions and generates multi-stage execution plans. The framework adapts to any algorithm or structured reasoning problem, with the current experiment focusing on causal discovery using the Peter-Clark (PC) algorithm.

## 🎯 Project Overview

**Core Capability**: Algorithm-agnostic planning and execution system that:
- Automatically detects algorithms mentioned in task descriptions
- Retrieves canonical mathematical knowledge for detected algorithms
- Generates appropriate multi-stage execution plans
- Executes plans with universal context flow management
- Falls back to generic reasoning when no specific algorithm is detected

**Current Experiment**: Causal discovery with Peter-Clark (PC) algorithm for hypothesis validation (PC algorithm is only mentioned in the task description in `main.py` - the rest of the system is completely generic).

## 🏗️ Architecture

### Core Components

- **`src/main.py`**: Entry point and experiment runner
  - **PC algorithm only mentioned here** in task description (lines 37-49)
  - Loads dataset samples and orchestrates complete workflow
  - Contains experiment-specific logic for current PC use case

- **`src/planner.py`**: **Algorithm-agnostic** planning engine
  - `detect_algorithm()`: Analyzes any task description for algorithm mentions
  - `retrieve_algorithm_knowledge()`: Fetches canonical stages for any detected algorithm
  - `create_planner()`: Returns algorithm-aware or generic planner as needed
  - `refine_schema()`: Automatically improves generic JSON schemas

- **`src/executor.py`**: **Universal** execution engine
  - `run_stage()`: Executes individual stages with context validation
  - `run_plan()`: Sequential execution of complete plans
  - Works with any algorithm or reasoning workflow

- **`src/models.py`**: **Generic** data structures
  - `Stage`: Universal stage definition (reads, writes, prompt, schema)
  - `Plan`: Complete execution plan with any number of stages

## 🔄 Algorithm-Agnostic Workflow

1. **Load Task**: Read task description and input data
2. **Algorithm Detection**: Automatically detect any mentioned algorithm
3. **Knowledge Retrieval**: Fetch canonical mathematical stages (if algorithm detected)
4. **Plan Generation**: Create algorithm-specific or generic reasoning plan
5. **Schema Refinement**: Automatically improve generic schemas
6. **Sequential Execution**: Run stages with universal context flow
7. **Result Extraction**: Return final result in expected format

## 🚀 Setup & Usage

### Prerequisites

```bash
# Install dependencies (inferred from code)
pip install pydantic-ai pandas python-dotenv
```

### Environment Configuration

Create `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

**⚠️ Security Note**: The current `.env` contains an exposed API key. Consider adding `.env` to `.gitignore`.

### Current PC Experiment

```bash
cd src/
python main.py  # Runs single sample experiment
```

**Data Requirements** (for current PC experiment):
- Expects `../data/test_dataset.csv` with columns:
  - `input`: Natural language premise and hypothesis
  - `label`: Expected boolean result
  - `num_variables`: Number of variables in the causal structure
  - `template`: Template identifier

## 🔧 Extending to Other Algorithms

The framework is designed to work with **any algorithm or structured reasoning problem**. To adapt:

### 1. Change Task Description Only
In `src/main.py`, modify the `task_description` (lines 37-49):
```python
task_description = """
Task: [Your new algorithm/reasoning task]
Algorithm: [Mention specific algorithm name for auto-detection]
Input available in context: [describe input format]
Expected output: [describe expected output format]
"""
```

### 2. Update Data Format (if needed)
- Modify `fetch_sample()` to load your data format
- Update input context in `run_complete_workflow()`

### 3. Everything Else Adapts Automatically
- Algorithm detection works for any algorithm name
- Knowledge retrieval fetches appropriate canonical stages
- Planning generates algorithm-specific or generic stages
- Execution engine handles any workflow
- Schema refinement works for any domain

## 🧠 Key Design Principles

### Algorithm-Agnostic Design
- **Universal execution engine**: Same executor for any algorithm
- **Automatic adaptation**: Framework detects and adapts to mentioned algorithms
- **Generic fallback**: Works with general reasoning when no algorithm detected
- **Domain-independent**: Context management works across all domains

### Adaptive Planning
- **Algorithm-aware**: Creates stages matching canonical mathematical phases
- **Knowledge-informed**: Uses retrieved algorithmic knowledge for planning
- **Schema-flexible**: Automatically refines output schemas for any domain
- **Context-validated**: Universal reads/writes validation

### Robust Execution
- **Sequential stages**: Each stage builds on previous context
- **Error handling**: Comprehensive validation and error reporting
- **Progress tracking**: Detailed logging with execution times and output previews
- **JSON validation**: Strict schema compliance for all outputs


## 🎛️ Configuration

- **Primary model**: `openai:o3-mini` (planning and execution)
- **Detection model**: `openai:gpt-4o-mini` (algorithm detection)
- Models can be changed in respective Agent constructors