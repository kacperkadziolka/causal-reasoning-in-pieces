# Causal Discovery Pipeline

## Running the Script

**Important:** Always run from the project root directory with `PYTHONPATH` set:

```bash
cd "/home/kacperkadziolka/University/Master's thesis"
PYTHONPATH="$(pwd)" python causal_discovery/main.py [options]
```

```bash
cd "/home/kacperkadziolka/University/Master's thesis"
  PYTHONPATH="/home/kacperkadziolka/University/Master's thesis" /home/kacperkadziolka/miniconda3/envs/local_env/bin/python causal_discovery/main.py
```

## Usage Examples

**OpenAI Backend:**
```bash
PYTHONPATH="$(pwd)" python causal_discovery/main.py \
  --backend openai \
  --mode sequential \
  --input_file data_peturbations/test_dataset_variable_refactorization.csv \
  --num_experiments 10
```

**DeepSeek Backend:**
```bash
PYTHONPATH="$(pwd)" python causal_discovery/main.py \
  --backend deepseek \
  --mode batched \
  --batch_size 8 \
  --num_experiments 20
```

**HuggingFace Backend:**
```bash
PYTHONPATH="$(pwd)" python causal_discovery/main.py \
  --backend huggingface \
  --mode batched \
  --batch_size 4 \
  --num_experiments 5
```

## Options

- `--backend`: Choose LLM backend (openai, huggingface, deepseek)
- `--mode`: Processing mode (sequential, batched)
- `--input_file`: Path to input CSV file
- `--num_experiments`: Number of experiments to run
- `--batch_size`: Batch size for batched mode
- `--debug`: Enable debug logging

## Requirements

- Set `PYTHONPATH` to project root
- API keys in `.env` file (OPENAI_API_KEY, DEEPSEEK_API_KEY, HF_TOKEN)
- Run from project root directory