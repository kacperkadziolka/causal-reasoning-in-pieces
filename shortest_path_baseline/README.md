# Shortest Path Baseline

Single-prompt baseline evaluating LLMs on the NLGraph shortest path task. Uses the OpenAI API (`o3-mini`) with different prompting strategies to solve weighted shortest path problems in a single reasoning step.

## Running the Script

```bash
cd "/path/to/project-root"
python shortest_path_baseline/main.py [options]
```

## Usage Examples

**Chain-of-Thought prompting:**
```bash
python shortest_path_baseline/main.py \
  --backend openai \
  --prompt_type cot_prompt \
  --num_experiments 100
```

**Algorithm-guided (Dijkstra) prompting:**
```bash
python shortest_path_baseline/main.py \
  --backend openai \
  --prompt_type algorithm_prompt \
  --difficulty easy
```

**Build-a-Graph prompting:**
```bash
python shortest_path_baseline/main.py \
  --backend openai \
  --prompt_type bag_prompt \
  --difficulty hard \
  --num_experiments 50
```

## Options

- `--backend`: LLM backend (currently `openai`)
- `--prompt_type`: Prompting strategy (`direct_prompt`, `cot_prompt`, `algorithm_prompt`, `bag_prompt`)
- `--difficulty`: Filter by difficulty (`easy`, `hard`, or omit for all)
- `--num_experiments`: Number of samples to run (omit for full dataset)
- `--input_file`: Path to NLGraph JSON (default: `../data/nlgraph_shortest_path_main.json`)
- `--debug`: Enable debug logging

## Prompt Strategies

| Strategy | Description |
|----------|-------------|
| `direct_prompt` | Ask the question directly with no guidance |
| `cot_prompt` | Chain-of-thought reasoning with answer format instruction |
| `algorithm_prompt` | Step-by-step Dijkstra algorithm walkthrough |
| `bag_prompt` | Build-a-Graph: construct graph structure before solving |

## Evaluation Metrics

- **Weight accuracy**: Whether the model's total weight matches the ground truth (primary metric)
- **Path validity**: Whether the model's path exists in the graph with correct edge weights
- **Extraction failure rate**: How often the answer could not be parsed from the model output

Metrics are reported overall and split by difficulty (easy/hard).

## Requirements

- API key in `.env` file (`OPENAI_API_KEY`)
- NLGraph shortest path dataset in `data/`
