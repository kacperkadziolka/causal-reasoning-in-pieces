# Shortest Path Pipeline

Multi-stage pipeline for the NLGraph shortest path task. Decomposes the problem into specialized LLM stages (graph parsing, Dijkstra execution, optional result verification), reusing the generic pipeline infrastructure from `causal_discovery/`.

## Running the Script

**Important:** Always run from the project root directory with `PYTHONPATH` set:

```bash
cd "/path/to/project-root"
PYTHONPATH="$(pwd)" python shortest_path/main.py [options]
```

## Usage Examples

**OpenAI Backend (batched):**
```bash
PYTHONPATH="$(pwd)" python shortest_path/main.py \
  --backend openai \
  --mode batched \
  --batch_size 4 \
  --num_experiments 380
```

**DeepSeek Backend (sequential):**
```bash
PYTHONPATH="$(pwd)" python shortest_path/main.py \
  --backend deepseek \
  --mode sequential \
  --num_experiments 50
```

**Filter by difficulty:**
```bash
PYTHONPATH="$(pwd)" python shortest_path/main.py \
  --backend openai \
  --mode batched \
  --difficulty hard \
  --num_experiments 200
```

**Skip verification stage (2 stages instead of 3):**
```bash
PYTHONPATH="$(pwd)" python shortest_path/main.py \
  --backend openai \
  --mode batched \
  --skip_verification
```

**Retry failed samples from a previous run:**
```bash
PYTHONPATH="$(pwd)" python shortest_path/main.py \
  --backend openai \
  --mode batched \
  --retry-file shortest_path/logs/failed_samples_20260219.json
```

## Options

- `--backend`: LLM backend (`openai`, `huggingface`, `deepseek`)
- `--mode`: Processing mode (`sequential`, `batched`)
- `--batch_size`: Batch size for batched mode (default: 4)
- `--num_experiments`: Number of samples to run (default: 380)
- `--difficulty`: Filter by difficulty (`easy`, `hard`, or omit for all)
- `--skip_verification`: Skip the result verification stage
- `--retry-file`: JSON file with failed sample IDs to retry
- `--seed`: Random seed (default: 42)
- `--input_file`: Path to NLGraph JSON (default: `data/nlgraph_shortest_path_main.json`)
- `--debug`: Enable debug logging

## Pipeline Stages

```
Question (natural language graph description)
    |
    v
[1] GraphParsingStage
    Parses the NLGraph text into a structured JSON representation
    (nodes, adjacency list, source, target)
    |
    v
[2] DijkstraExecutionStage
    Executes Dijkstra's algorithm step-by-step on the parsed graph
    (path, total_weight)
    |
    v
[3] ResultVerificationStage (optional)
    Verifies path connectivity, weight accuracy, and optimality
    (corrected path/weight if needed)
    |
    v
Evaluation against ground truth
```

## Evaluation Metrics

- **Weight accuracy**: Whether the model's total weight matches the ground truth (primary metric)
- **Path validity**: Whether the model's path exists in the graph with correct edge weights
- **Path-weight consistency**: Whether the path weight matches the claimed total weight
- **Extraction failure rate**: How often the answer could not be parsed

Metrics are reported overall and split by difficulty (easy/hard).

## Requirements

- Set `PYTHONPATH` to project root
- API keys in `.env` file (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `HF_TOKEN`)
- NLGraph shortest path dataset in `data/`
