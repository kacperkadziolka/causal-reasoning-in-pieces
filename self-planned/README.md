# Causal Reasoning in Pieces — **LLM‑Planned Static Workflow**

*A generic, domain‑agnostic “Plan → Execute” pipeline where the LLM invents the stages, schemas, and prompts — and then executes them to answer a task.*

---

## Summary

This document describes a **static, two‑pass workflow** that keeps engineering overhead minimal while maximizing **LLM‑determined structure**:

1. **Plan (LLM)** — Given a task description and sample inputs, the LLM returns a **Plan JSON** containing an arbitrary number of **stages**. Each stage defines:

   * what it **reads** from a shared context,
   * what it **writes** back to that context,
   * a concise **prompt template** to produce the next artifact,
   * a strict **JSON `output_schema`** for its result,
   * optional **LLM self‑validation rubrics** (no hand‑written validators required).

2. **Execute (LLM)** — A tiny **deterministic runner** loops through the stages. For each stage it renders the prompt with the minimal context slice, calls the LLM to return JSON that matches the declared schema, optionally asks the LLM to self‑validate/repair, writes outputs into context, and proceeds. The final answer is read from a context key defined by the plan (e.g., `"decision"`).

**Key properties**

* **Fully LLM‑determined decomposition** — You do **not** fix the number or names of stages.
* **No domain validators to implement** — Optional validation is performed by the **LLM itself** using a rubric it provides in the plan.
* **Domain‑agnostic** — To switch domains, change only the one‑line method hint in the task description (e.g., “use the PC algorithm” → “use FCI” → “prove via induction”).

---

## Goals & Non‑Goals

**Goals**

* Let the LLM **decide** the workflow structure and I/O schemas per task (or per dataset “regime”).
* Keep the orchestrator **small, deterministic, and reusable**.
* Enable **replicable** runs by logging the exact Plan JSON, seeds/temperatures, and model names.

**Non‑Goals**

* Implementing causal‑discovery routines (CI tests, Meek, MEC checks) or any domain‑specific validators.
* Hard‑coding a fixed number of stages or their semantics.

---

## High‑Level Flow

```
┌──────────┐        ┌──────────────┐        ┌─────────────┐
│  Inputs  │  ──▶   │   Planner    │  ──▶   │   Plan JSON │
│ (task +  │        │    (LLM)     │        │ (stages...) │
│  sample) │        └──────────────┘        └─────────────┘
     │                                         │
     │                                         ▼
     │                                  ┌─────────────┐
     └────────────────────────────────▶ │  Executor   │───▶ Final answer
                                        │ (tiny loop) │
                                        └─────────────┘
```

**Terminology**

* **Context** — a JSON dictionary threaded through stages (created on the fly). The Plan defines which keys appear by declaring `writes`.
* **Stage** — a unit of work chosen by the LLM; it reads a subset of the context, emits JSON that matches its own `output_schema`, and writes selected keys back.

---

## Plan JSON (meta‑schema)

> The LLM invents the content; you only fix the *envelope* so the runner can execute it.

```json
{
  "stages": [
    {
      "id": "string",
      "goal": "string",
      "reads": ["context_key_1", "context_key_2"],
      "writes": ["new_key_1", "new_key_2"],
      "prompt_template": "string with {placeholders} for reads",
      "output_schema": { "type": "object", "properties": { }, "required": [ ] },
      "n": 1,
      "select": "first | vote | best_of",
      "validation": {
        "validator_prompt": "optional natural-language rubric to self-check",
        "accept_if": "optional NL acceptance rule",
        "repair_strategy": "optional NL suggestion for how to regenerate"
      }
    }
  ],
  "aggregation": {
    "policy": "optional description of how to combine n samples (e.g., majority)"
  },
  "final_key": "name-of-context-key-to-return-as-answer",
  "abstain_policy": "optional conditions to output 'Undetermined'"
}
```

* **`reads` / `writes`** define the **shared context** contract.
* **`output_schema`** is a JSON Schema the **LLM commits to**; the runner simply parses JSON.
* **`n` / `select`** enable basic self‑consistency (e.g., majority vote).

---

## Example Sample (CORR2CAUSE‑style)

**Task**
“Given the *Premise* and the *Hypothesis*, decide whether the hypothesis is **True**, **False**, or **Undetermined** **using the PC algorithm**.”

**Inputs**

```json
{
  "premise": "Suppose there is a closed system of 2 variables, A and B. All the statistical relations among these 2 variables are as follows: A correlates with B.",
  "hypothesis_text": "B causes something else which causes A."
}
```

> The LLM will produce its own multi‑stage plan. A plausible outcome for this sample is `{"decision": "False"}` (a closed 2‑variable universe has no mediator), but the **stage decomposition** is entirely LLM‑chosen.

---

## Minimal Runner (Option A — **PydanticAI**)

> One model, two phases. No domain validators. The LLM supplies the Plan and produces JSON per stage.

**Install**

```bash
pip install pydantic-ai openai
```

**`src/runner.py`**

```python
from typing import Any, Dict, List
from pydantic import BaseModel
from pydantic_ai import Agent

# ---------- Models ----------
class Stage(BaseModel):
    id: str
    goal: str | None = None
    reads: List[str] = []
    writes: List[str]
    prompt_template: str
    output_schema: Dict[str, Any]
    n: int | None = 1
    select: str | None = "first"
    validation: Dict[str, Any] | None = None

class Plan(BaseModel):
    stages: List[Stage]
    aggregation: Dict[str, Any] | None = None
    final_key: str | None = None
    abstain_policy: str | None = None

# ---------- Agents ----------
planner = Agent(
    system_prompt=(
        "You are a planning model. Given a task description and inputs, "
        "output ONLY a JSON object with fields: stages[], optional aggregation, "
        "optional final_key, optional abstain_policy. "
        "Each stage has id, reads, writes, prompt_template, output_schema, "
        "and optional n/select/validation. "
        "Do NOT assume any domain validators or tools; rely on your own reasoning."
    ),
    result_type=Plan,
    model="openai:gpt-4.1-mini",
    temperature=0
)

executor = Agent(
    system_prompt=(
        "You produce STRICT JSON outputs for a given stage. "
        "The user will provide a prompt_template rendered with context. "
        "Return ONLY JSON matching the given output_schema."
    ),
    model="openai:gpt-4.1-mini",
    temperature=0
)

# ---------- Helpers ----------
def _select(samples: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    mode = (mode or "first").lower()
    if mode == "first" or len(samples) == 1:
        return samples[0]
    if mode == "vote":
        from collections import Counter
        import json
        enc = [json.dumps(s, sort_keys=True) for s in samples]
        chosen_enc, _ = Counter(enc).most_common(1)[0]
        return json.loads(chosen_enc)
    # best_of placeholder → first
    return samples[0]

def call_stage(stage: Stage, context: Dict[str, Any]) -> Dict[str, Any]:
    read_view = {k: context.get(k) for k in stage.reads}
    rendered = stage.prompt_template.format(**read_view)
    # Include the schema in the message to steer JSON shape
    message = f"{rendered}\n\n# Schema:\n{stage.output_schema}"
    samples: List[Dict[str, Any]] = []
    k = stage.n or 1
    for _ in range(k):
        out = executor.run(message).data
        if not isinstance(out, dict):
            raise RuntimeError("Stage must return a JSON object.")
        samples.append(out)
    return _select(samples, stage.select or "first")

def run_plan(plan: Plan, context: Dict[str, Any]) -> Dict[str, Any]:
    ctx = dict(context)
    for stage in plan.stages:
        out = call_stage(stage, ctx)
        # Write declared keys
        for key in stage.writes:
            if key not in out:
                raise RuntimeError(f"Stage '{stage.id}' did not write required key '{key}'.")
            ctx[key] = out[key]
    return ctx

if __name__ == "__main__":
    TASK = (
        "Task: Given the premise and the hypothesis, decide whether the hypothesis "
        "is True, False, or Undetermined using the PC algorithm."
    )
    PREMISE = (
        "Suppose there is a closed system of 2 variables, A and B. "
        "All the statistical relations among these 2 variables are as follows: "
        "A correlates with B."
    )
    HYP = "B causes something else which causes A."

    plan = planner.run(
        f"{TASK}\n\nInputs available in context: 'premise', 'hypothesis_text'. "
        "You choose any number of stages. Keep prompts concise and JSON-only outputs."
    ).data

    context_in = {"premise": PREMISE, "hypothesis_text": HYP}
    final_ctx = run_plan(plan, context_in)

    final_key = plan.final_key or "decision"
    print("FINAL:", final_ctx.get(final_key, final_ctx))
```

**Usage**

```bash
python src/runner.py
```

> To **reuse a plan** for many samples, serialize `plan.model_dump_json()` to `plans/<name>.json` and reload it instead of calling the planner each time.

---

## Minimal Runner (Option B — **OpenAI Responses API**)

If you prefer not to use PydanticAI, mirror the logic above with OpenAI’s Responses API and `response_format={"type":"json_object"}` (or `json_schema`). Implement two calls:

1. **Plan call** → obtain Plan JSON.
2. **Stage call(s)** → for each stage, send the rendered prompt and parse JSON.

*(The overall control flow is identical to the PydanticAI version.)*

---

## Configuration & Reproducibility

**Config knobs**

* `model`, `temperature`, `max_tokens` (plan vs execute can differ).
* Stage‑level `n` (self‑consistency) and `select` policy.
* Simple plan cache keyed by a hash of the task description.

**Log for every run**

* Plan JSON, model name(s), temperatures, seeds (if supported), exact prompts per stage, and the final context.

---

## Evaluation Guidance

* **Correctness** on your dataset: accuracy/F1 for `{True, False, Undetermined}` (or your domain’s target).
* **Efficiency**: total tokens and wall‑time per sample (Plan + Execute).
* **Robustness ablations**

  * **Plan reuse vs per‑sample plan** (hybrid often wins: one plan per “regime”).
  * **Self‑consistency**: `n=1` vs `n=3/5` and `select=vote`.
  * **Single‑model vs two‑model** (e.g., same model for planning/execution vs smaller model for execution).

---

## Switching Domains

This pipeline is **domain‑agnostic**. To reuse in another area, change only the **task description** line, e.g.:

* “Using the PC algorithm” → “Using FCI” → “Using rule‑based logic” → “Using dynamic programming”.
* Keep everything else identical. The LLM will invent appropriate stages, keys, and schemas.

---

## FAQ

**Do I need domain validators (acyclicity, CPDAG, CI tests, etc.)?**
No. In this static workflow, **all** decomposition and execution is performed by the LLM. If you later want extra rigor, you can add LLM‑based self‑validation rubrics (judge prompts) without writing domain code.

**Should I plan per sample or once?**
Start by **planning once per dataset regime** and reusing the plan; re‑plan only when inputs shift materially (e.g., from “closed system” tasks to “open world” tasks).

**Can I access intermediate artifacts?**
Yes — any keys the plan writes (e.g., `variables`, `constraints`, `skeleton`, `cpdag`) remain in the final context.

---

## Suggested Repository Structure

```
.
├─ docs/
│  └─ LLM_Planned_Static_Workflow.md   # this file
├─ src/
│  ├─ runner.py                        # Option A (PydanticAI)
│  └─ runner_responses.py              # Option B (Responses API) [optional]
├─ plans/
│  └─ pc_default_plan.json             # cached plan(s) per regime [optional]
├─ data/
│  └─ examples.jsonl                   # (premise, hypothesis, label) for eval
└─ README.md
```

---

## Ready‑to‑Use **Planner Prompt** (generic)

```
You are a planning model. Given a task description and inputs, return ONLY a JSON object with:
- stages[]: each stage has id, reads[], writes[], prompt_template (concise), output_schema (strict JSON Schema),
  and optional n/select/validation.
- optional aggregation, optional final_key, optional abstain_policy.

Guidelines:
- Pick ANY number of stages; you decide the decomposition.
- Keep input/output JSON compact and domain‑agnostic (arrays/objects, no prose).
- Do NOT assume any domain validators or tools; plan outputs the model itself can produce.
- Prefer minimal context per stage (only refer to the keys listed in reads).
- Ensure every writes[] key is actually produced by the stage.

Task: {YOUR_TASK_DESCRIPTION_HERE}

Inputs available in context: {list them, e.g., "premise", "hypothesis_text"}.

Constraints: prompts must be short; outputs MUST match their schemas exactly; avoid unnecessary text.
```

---

### License / Citation Note

If you reuse this documentation, please include a reference to the repository and the paper “Causal Reasoning in Pieces: Modular In‑Context Learning for Causal Discovery.”
