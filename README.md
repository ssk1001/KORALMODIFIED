# KORAL (Stage II): Knowledge-Graph Guided LLM Reasoning for SSD Operational Analysis

This repository contains **KORAL**, a pipeline that combines:
- **SSD telemetry** (SMART + optional workload / environment / controller context),
- a **Data Knowledge Graph (DataKG)** built from each sample,
- a **Literature Knowledge Graph (LitKG)** (Turtle/TTL) used to retrieve domain evidence,
- and an **LLM** (GPT‑4o) to produce **Predictive**, **Descriptive**, **Prescriptive**, and **What‑If** analyses.

Stage II is designed to be **modular**: each modality (SMART, workload, environment, flash type, controller policies) has its own feature module, and evaluation is computed automatically per task.

---

## Repository structure

Recommended layout (repo root):

```
dataset/
  alibaba/
    smartlog2018ssd/               # daily logs (raw)
    smartlog2019ssd/               # daily logs (raw)
    selected_attributes/           # intermediate output (optional)
    MA1/2018/ ... MA1/2019/ ...     # filtered per-model daily CSVs
    ... (MB1, MB2, MC1, MC2)
    test_data/                     # 30-day windows + labels (Stage II input)
  google/
    raw_data/
      badchip.csv
      swaplog.csv
      errorlog.csv
    test_data/                     # 30-day windows + labels (Stage II input)
  env/
    env_effects.csv                # environment-only ground-truth effects (paper-derived)
    fio_workloads/                 # *.fio job files (generated)
data_preparation/                  # scripts for filtering + windowing
stage_II/                          # Stage II modular package (this repo)
scripts/
```

> **Note**: If your env file is currently named differently (e.g., `env_paper_only_effects.csv`), copy/rename it to:
`dataset/env/env_effects.csv`.

---

## Dependencies

Minimal requirements for Stage II:

```bash
pip install pandas numpy requests
```

Optional (recommended) for richer KG handling:
```bash
pip install rdflib
```

---

## Inputs required by Stage II

Stage II expects **(A)** an input CSV with samples and **(B)** KG/rule assets:

### A) Input CSV (your sampled test set)
Your Stage II input CSV can be any of the dataset “types” you create (SMART-only, SMART+workload, SMART+env, etc.).  
Stage II treats any column matching `r_<number>` (e.g., `r_5`, `r_233`) as a SMART attribute.

Recommended columns (where available):
- `sample_id` (optional) — if missing, Stage II auto-creates IDs
- `disk_id` (optional)
- `ds` (date string; optional)
- `failure` or `label` — ground truth (0/1)
- `ttf_days` (optional regression GT)
- `tail_latency_ms` (optional regression GT)
- `app` (Alibaba workload tag; optional)
- environment columns (from `env_effects.csv`, optional)
- `flash_type` (optional)
- `algorithms`/`policies`/`controller_policies` (optional)

SMART values:
- can be scalars **or** a 30-day series (e.g., a JSON list string like `"[1,2,3,...]"`).

### B) Assets (rules, taxonomy, literature KG)
Place these files at repo root (or update paths in `stage_II/config.py`):
- `rule_base.json` — curated rule set used by Stage I / DataKG logic
- `taxonomy.json` — concept taxonomy
- `global_knowledge_graph.ttl` — Literature KG (LitKG) used for retrieval grounding
- `ssd_cot_prompt.txt` — optional prompt guidance for analysis style/format

---

## Data preparation: end-to-end overview

Stage II assumes you already produced **windowed, labeled samples**.

### 1) Alibaba: filter + select SMART attributes (per model)
- Filter raw daily logs by models: `MB1, MB2, MA1, MA2, MC1, MC2`
- Select the 19 SMART attributes + `disk_id`, `ds`
- Drop model-specific missing attributes
- Save per-model daily CSVs to:
  - `dataset/alibaba/MB1/2018`, `dataset/alibaba/MB1/2019`, etc.

(See your script: `data_preparation/final_drop_missing_by_model.py`.)

### 2) Alibaba: build 30‑day windows + labels (failure-based)
- Use `ssd_failure_tag.csv` and failure time to extract 30‑day windows
- Split healthy:failed = 70:30
- Output to: `dataset/alibaba/test_data/*.csv`

(See: `data_preparation/build_test_data_windows.py`.)

### 3) Google: build 30‑day windows + labels
- Input: `badchip.csv`, `swaplog.csv`, `errorlog.csv`
- Output: `dataset/google/test_data/*.csv`

(See: `data_preparation/build_google_test_data_windows.py`.)

### 4) Environment effects (paper-derived)
- Use: `dataset/env/env_effects.csv`
- This file contains **environment conditions → reported performance deltas** (from papers), not 30-day windows.

### 5) Workload (fio)
- Generate a workload suite:
  ```bash
  python data_preparation/generate_fio_workloads.py     --out_dir dataset/env/fio_workloads     --filename /dev/nvme0n1     --runtime 120 --time_based     --size 8G
  ```
- Or use a single job file like `fit_sample.fio`.

⚠️ Many fio jobs write data. Use a safe test device or read-only workloads.

---

## Generating Stage II dataset “types” (pairs/triples)

Use your dataset generator to produce the dataset types described in Table 1 (examples):

```bash
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART --n 1000 --smart-source alibaba --out dataset/stage2/smart_only.csv
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_ENV --n 500 --smart-source alibaba --env-csv dataset/env/env_effects.csv --out dataset/stage2/smart_env.csv
python stage_II/stage2_pair_dataset_generator.py --dataset-type ENV_WORKLOAD --n 500 --fio-path dataset/env/fio_workloads --out dataset/stage2/env_fio.csv
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_FT --n 500 --smart-source alibaba --out dataset/stage2/smart_ft.csv
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_AL --n 500 --smart-source alibaba --out dataset/stage2/smart_al.csv
```

---

## Stage II: running the full modular pipeline

### 1) Set your OpenAI key
```bash
export OPENAI_API_KEY="sk-..."
```

### 2) Run Stage II
From repo root:

```bash
python -m stage_II.cli   --dataset_type SMART_ALIBABA   --input_csv dataset/stage2/smart_only.csv   --tasks predictive,descriptive,prescriptive,whatif   --limit_rows 50
```

Other examples:

**SMART + Env + Workload**
```bash
python -m stage_II.cli   --dataset_type SMART_ENV_WORKLOAD   --input_csv dataset/stage2/smart_env_workload.csv   --tasks predictive,descriptive,prescriptive,whatif   --limit_rows 100
```

**Environment-only (paper effects)**
```bash
python -m stage_II.cli   --dataset_type ENV   --input_csv dataset/env/env_effects.csv   --tasks descriptive,whatif   --limit_rows 100
```

---

## Stage II outputs

Each run writes to:

```
stage_II/runs/<RUN_NAME>/
  input_samples.csv
  responses.jsonl
  metrics_per_sample.csv
  metrics_summary.json
  data_kg_ttl/<sample_id>.ttl    # if rdflib installed
```

### `responses.jsonl`
Each line contains:
- `sample_id`
- `task`
- `prompt_terms` (used for LitKG retrieval)
- `response_text` (raw model output)
- `response_json` (parsed JSON, if valid)

---

## Evaluation metrics (Stage II)

Stage II computes metrics depending on the task and what ground truth is available in your CSV.

### 1) Predictive (classification)
If your input CSV includes `failure` or `label`:
- **Precision, Recall, Accuracy**
- Confusion counts: TP/FP/FN/TN

Optional regression metrics if you include ground truth:
- `TTF_MSE` (if `ttf_days` exists and model predicts it)
- `TL_MSE` (if `tail_latency_ms` exists and model predicts it)

### 2) Descriptive & Prescriptive (text + grounding)
- **B4**: BLEU‑4 overlap vs reference text (requires `ref_descriptive` / `ref_prescriptive`)
- **RL**: ROUGE‑L F1 vs reference text
- **FiP**: Faithfulness‑in‑Prompting precision  
  Computed as: supported atomic claims / total atomic claims  
  Support is validated by **reference IDs** included in the response JSON.

### 3) What‑If (counterfactual validity)
- **B4 / RL** if `ref_whatif` exists
- **CFV**: Counterfactual validity (direction/evidence consistency)  
  In the current implementation, CFV uses a conservative check: each statement must have evidence and a clear direction; you can tighten this by adding an evidence→direction lookup derived from env/lit KG.

---

## Prompted tasks (what Stage II asks the LLM to produce)

Stage II issues one prompt per sample per task, and requires **strict JSON output**.

### Predictive output (example schema)
```json
{
  "task": "predictive",
  "sample_id": "s12",
  "predicted_failure": 1,
  "predicted_ttf_days": 7,
  "predicted_tail_latency_ms": null,
  "rationale": "Rising error indicators and a recent changepoint suggest imminent risk.",
  "atomic_claims": [
    {"claim": "Attribute r_233 shows a significant changepoint.", "support": ["IR:AF_r_233"]}
  ]
}
```

### Descriptive output (example schema)
```json
{
  "task": "descriptive",
  "sample_id": "s12",
  "summary": "SMART indicators suggest moderate wear and elevated error risk.",
  "key_risks": ["increasing error-related SMART", "recent instability"],
  "atomic_claims": [
    {"claim": "r_5 median increased over the window.", "support": ["IR:AF_r_5"]},
    {"claim": "Humidity can degrade tail latency post-impact in TLC.", "support": ["LIT_3"]}
  ]
}
```

### Prescriptive output (example schema)
```json
{
  "task": "prescriptive",
  "sample_id": "s12",
  "recommendations": [
    {
      "action": "Increase monitoring cadence and migrate hot data off the drive.",
      "priority": "high",
      "justification": "Recent instability + error indicators increase failure risk.",
      "support": ["IR:AF_r_5", "IR:AF_r_187", "LIT_2"]
    }
  ],
  "atomic_claims": [
    {"claim": "Drive shows elevated error-related SMART statistics.", "support": ["IR:AF_r_187"]}
  ]
}
```

### What‑if output (example schema)
```json
{
  "task": "whatif",
  "sample_id": "s12",
  "scenario": "If inlet temperature decreases by 5°C and RH decreases by 10%, how do tail latency and failure risk change?",
  "analysis": "Lower humidity generally reduces post-impact latency risk under high-RH conditions.",
  "counterfactual_statements": [
    {
      "statement": "Reducing RH should decrease tail-latency risk if the current condition includes high humidity.",
      "variable": "relative_humidity_pct",
      "delta": -10,
      "effect": "tail_latency_change_pct",
      "effect_direction": "decrease",
      "evidence": ["ENV:condition_7"]
    }
  ]
}
```

---

## Ontology / DataKG examples

Stage II builds a lightweight **DataKG** per sample (TTL output if `rdflib` is installed).  
This is **not** the LitKG; it is sample-specific.

### Core node types (examples)
- `ex:Sample`
- `ex:AttributeFrame` (SMART attribute summary)
- `ex:Env` / `ex:Workload` / `ex:FlashType` / `ex:Algorithms` (optional modality nodes)

### Key relation types (examples)
- `ex:hasAttributeFrame` — Sample → AttributeFrame
- `ex:has_env` — Sample → Env condition node
- `ex:has_workload` — Sample → Workload node
- `ex:has_flash_type` — Sample → FlashType node
- `ex:has_algorithms` — Sample → controller policies node
- `ex:attribute`, `ex:median`, `ex:p95`, `ex:slope`, `ex:changepoint_idx`, `ex:outliers` — AttributeFrame properties

### Minimal TTL snippet
```turtle
@prefix ex: <http://example.org/koral-data#> .

ex:s12 a ex:Sample ;
  ex:hasAttributeFrame ex:AF_r_233 ;
  ex:has_workload ex:WL_APP ;
  ex:has_env ex:ENV_1 .

ex:AF_r_233 a ex:AttributeFrame ;
  ex:attribute "r_233" ;
  ex:median "12.0" ;
  ex:p95 "31.0" ;
  ex:slope "0.42" ;
  ex:changepoint_idx "18" ;
  ex:outliers "2" .

ex:WL_APP a ex:Workload ;
  ex:type "app_tag" ;
  ex:value "OLTP" .

ex:ENV_1 a ex:Env ;
  ex:temperature_c "60" ;
  ex:relative_humidity_pct "80" .
```

> In Stage II prompts, LLM outputs are required to cite evidence using **reference IDs**
(e.g., `IR:AF_r_233`, `ENV:condition_7`, `LIT_3`).  
Those IDs are validated during metric computation.

---

## Stage II modular code map

```
stage_II/
  cli.py                     # CLI entrypoint
  pipeline.py                # orchestration: IR → DataKG → LitKG → LLM → metrics
  config.py
  features/
    smart.py                 # SMART parsing + robust stats + changepoints
    env.py                   # environment parsing
    workload.py              # app tag + fio parsing
    flash_type.py
    algorithms.py
  kg/
    data_kg.py               # per-sample TTL materialization (optional)
    literature_kg.py         # retrieval from global_knowledge_graph.ttl
  llm/
    openai_client.py         # GPT-4o via Chat Completions endpoint
  prompts/
    templates.py             # task prompts (predictive / descriptive / prescriptive / whatif)
  evaluation/
    metrics_predictive.py    # P/R/A, MSE
    metrics_text.py          # BLEU-4, ROUGE-L
    grounding.py             # FiP, CFV
  utils/
    io.py, json_utils.py, text.py
```

---

## Troubleshooting

### “OPENAI_API_KEY is not set”
Export the key:
```bash
export OPENAI_API_KEY="sk-..."
```

### No `.ttl` files produced
Install rdflib:
```bash
pip install rdflib
```

### BLEU/ROUGE are zero
Your input CSV likely does not include reference text columns (`ref_descriptive`, `ref_prescriptive`, `ref_whatif`).  
This is expected if you are evaluating without gold summaries.

---

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{AkewarEtAl_IPDPS_2026,
  author    = {Akewar, Mayur and Madireddy, Sandeep and Luo, Dongsheng and Bhimani, Janki},
  title     = {KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis},
  booktitle = {IEEE International Parallel \& Distributed Processing Symposium (IPDPS)},
  year      = {2026},
  note      = {To appear}
}
```
