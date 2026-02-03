# KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis

KORAL is a two-stage pipeline:

- **Stage I (Literature → LitKG):** builds a **Literature Knowledge Graph (LitKG)** from SSD papers (PDF/TXT/MD) using a taxonomy-guided extraction prompt. Outputs **per-paper TTL/JSON**, updates **taxonomy.json** with new concepts, and maintains a **merged global_knowledge_graph.ttl**.
- **Stage II (Telemetry → DataKG → LLM):** builds a **Data Knowledge Graph (DataKG)** per sample from SSD telemetry (SMART + optional workload/environment/controller context), retrieves relevant literature evidence from LitKG, calls **GPT‑4o**, and evaluates outputs across multiple tasks.

Stage II is modular: each modality (SMART, workload, environment, flash type, controller policies) has its own feature module. Metrics are computed automatically per task.

---

## Repository structure

Recommended layout (repo root):

```
rule_base.json
taxonomy.json
global_knowledge_graph.ttl          # produced by Stage I (or provided)
prompts/
  ssd_cot_prompt.txt                # Stage I prompt (paper → KG)
  ssd_prompt_addenda_auto.txt        # optional, auto-augmented by Stage I

dataset/
  alibaba/
    smartlog2018ssd/                # raw daily logs (Alibaba)
    smartlog2019ssd/
    MA1/2018/ ... MA1/2019/ ...     # filtered per-model daily CSVs
    MB1/2018/ ... etc.
    test_data/                      # 30-day windows + labels (Stage II input)
    ssd_failure_tag.csv
  google/
    raw_data/
      badchip.csv
      swaplog.csv
      errorlog.csv
    test_data/                      # 30-day windows + labels (Stage II input)
  env/
    env_effects.csv                 # paper-derived env → performance effects
    fio_workloads/                  # generated *.fio workload configs

data_preparation/                   # scripts: filtering + windowing + dataset mixing
stage_II/                           # Stage II modular package
```

> If your environment file is currently named differently (e.g., `env_paper_only_effects.csv`), copy/rename it to  
`dataset/env/env_effects.csv`.

---

## Dependencies

### Stage I (LitKG builder)
```bash
pip install openai rdflib PyPDF2 python-dotenv
```

### Stage II (telemetry reasoning + evaluation)
```bash
pip install pandas numpy requests
```

Optional (recommended, enables TTL outputs):
```bash
pip install rdflib
```

---

## Stage I: Build / Update the Literature Knowledge Graph (LitKG)

Stage I converts SSD papers into KG triples aligned to **taxonomy.json**, while storing **evidence-backed assertions**.

### Inputs
- `taxonomy.json` (concept taxonomy; gets updated in-place if new concepts are proposed)
- A folder of papers: `*.pdf`, `*.txt`, or `*.md`
- `prompts/ssd_cot_prompt.txt` (and optional auto addenda)

### Output (in `--out_dir`)
- `*.ttl` (per-paper TTL)
- `*.kg.json` (per-paper extracted JSON)
- `global_knowledge_graph.ttl` (merged LitKG across runs; appended/merged, not overwritten)
- `taxonomy.json` may be updated with newly proposed concepts
- `prompts/ssd_prompt_addenda_auto.txt` may be extended to capture consistent mappings

### Run Stage I
```bash
export OPENAI_API_KEY="sk-..."

python ssd_kg_pipeline.py \
  --papers_dir papers/ssd \
  --taxonomy taxonomy.json \
  --out_dir outputs/stage_I \
  --model gpt-4o
```

To use the LitKG in Stage II, either:
- copy the generated `outputs/stage_I/global_knowledge_graph.ttl` to repo root, **or**
- point Stage II config to that path.

---

## Stage II: Telemetry → DataKG → Retrieval → GPT‑4o → Metrics

Stage II operates over a **sample CSV** (SMART-only or joined with workload/environment/controller context). It builds per-sample DataKG features, retrieves relevant evidence from the LitKG, prompts the LLM, and evaluates.

### Input CSV expectations
Stage II treats any column matching `r_<number>` (e.g., `r_5`, `r_233`) as a SMART attribute.

Recommended columns (depending on dataset type):
- `sample_id` (optional; auto-created if missing)
- `disk_id` (optional)
- `ds` (optional date string)
- `failure` or `label` (0/1 ground truth)
- `app` (Alibaba workload tag; optional)
- environment columns (from `env_effects.csv`; optional)
- `flash_type` (optional)
- `controller_policies` / `algorithms` (optional)

SMART values can be:
- scalars, or
- 30-day series encoded as a JSON list string (e.g., `"[1,2,3,...]"`).

### Running Stage II
```bash
export OPENAI_API_KEY="sk-..."

python -m stage_II.cli \
  --dataset_type SMART_ALIBABA \
  --input_csv dataset/stage2/smart_only.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --limit_rows 100
```

### Outputs
Each run writes to:

```
stage_II/runs/<RUN_NAME>/
  input_samples.csv
  responses.jsonl
  metrics_per_sample.csv
  metrics_summary.json
  data_kg_ttl/<sample_id>.ttl       # if rdflib installed
```

---

## Data preparation (Alibaba + Google + env + fio)

Stage II assumes you have produced **windowed, labeled samples**.

### Alibaba
1) **Filter daily logs by model**: MA1, MA2, MB1, MB2, MC1, MC2  
2) **Select 19 SMART attributes + `disk_id`, `ds`**  
3) **Drop model-specific missing attributes**  
4) **Build 30‑day windows based on `ssd_failure_tag.csv`** and label samples  
5) **Sample N records with 70:30 healthy:failed ratio**

### Google
1) Combine `badchip.csv`, `swaplog.csv`, and `errorlog.csv`  
2) Build labeled 30-day windows  
3) Sample N records with 70:30 healthy:failed ratio

### Environment effects (paper-derived)
Use `dataset/env/env_effects.csv` directly (no 30‑day windows). This file maps conditions → reported performance deltas.

### Workload (fio)
Generate or reuse fio configs in `dataset/env/fio_workloads/`.

---

## Generating dataset “types” (pairs/triples)

You can generate datasets matching Table 1 by mixing modalities.

Examples:

```bash
# SMART-only (Alibaba/Google)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART --n 1000 --smart-source alibaba --out dataset/stage2/smart_only.csv

# SMART + Env (no app)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_ENV --n 500 --smart-source alibaba --env-csv dataset/env/env_effects.csv --out dataset/stage2/smart_env.csv

# Env + Workload
python stage_II/stage2_pair_dataset_generator.py --dataset-type ENV_WORKLOAD --n 500 --fio-path dataset/env/fio_workloads --out dataset/stage2/env_fio.csv

# SMART + Flash Type (FT)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_FT --n 500 --smart-source alibaba --out dataset/stage2/smart_ft.csv

# SMART + Algorithms/Policies (AL)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_AL --n 500 --smart-source alibaba --out dataset/stage2/smart_al.csv
```

---

## Ontology examples

KORAL uses two related graphs:

### 1) LitKG (Stage I) — evidence-backed assertions from papers
LitKG stores extracted relationships as **reified assertions** with explicit provenance.

Key node type:
- `ns1:Assertion` (reified triple)

Key properties used in the LitKG:
- `ns1:subject`
- `ns1:predicate`
- `ns1:object`
- `ns1:confidence`
- `dcterms:source` (text evidence)

Minimal TTL snippet (from the global LitKG pattern):

```turtle
<http://example.org/ssd/assertion/002757e2fa2c4c1caa38b3fadd6e276f> a ns1:Assertion ;
    ns1:confidence 0.8 ;
    ns1:subject ns1:E8_WorkloadProfile_1 ;
    ns1:predicate ns1:hasAccessPattern ;
    ns1:object <http://example.org/ssd/taxonomy/SSD/EnvironmentalAndOperationalContext/WorkloadProfile/AccessPattern/Sequential> ;
    dcterms:source "Data can be read from and written to anywhere on the device, hence supporting random and sequential I/O operations." .
```

LitKG entity typing pattern:
- Entities are typed to taxonomy classes (URI in `http://example.org/ssd/taxonomy/...`)
- Entities use:
  - `rdfs:label`
  - `rdfs:isDefinedBy` (taxonomy class URI)

### 2) DataKG (Stage II) — per-sample telemetry graph
DataKG is sample-specific and captures time-window summaries and optional modalities.

Core node types (examples):
- `ex:Sample`
- `ex:AttributeFrame` (SMART attribute summary)
- `ex:Env`, `ex:Workload`, `ex:FlashType`, `ex:Algorithms` (optional)

Key relations (examples):
- `ex:hasAttributeFrame`
- `ex:has_env`, `ex:has_workload`, `ex:has_flash_type`, `ex:has_algorithms`
- `ex:median`, `ex:p95`, `ex:slope`, `ex:changepoint_idx`, `ex:outliers`

Minimal TTL snippet:

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
```

---

## Stage II task examples (LLM output schemas)

Stage II prompts the LLM to produce **strict JSON** and to cite evidence via **reference IDs**:
- Data evidence: `IR:AF_r_233` (derived from DataKG feature frames)
- Environment evidence: `ENV:condition_7` (from `env_effects.csv`)
- Literature evidence: `LIT_3` (from LitKG retrieval)

### Predictive (classification)
```json
{
  "task": "predictive",
  "sample_id": "s12",
  "predicted_failure": 1,
  "predicted_ttf_days": 7,
  "rationale": "Rising error indicators and a changepoint suggest imminent risk.",
  "atomic_claims": [
    {"claim": "Attribute r_233 shows a significant changepoint.", "support": ["IR:AF_r_233"]}
  ]
}
```

### Descriptive
```json
{
  "task": "descriptive",
  "sample_id": "s12",
  "summary": "SMART indicators suggest moderate wear and elevated error risk.",
  "key_risks": ["increasing error-related SMART", "recent instability"],
  "atomic_claims": [
    {"claim": "r_5 median increased over the window.", "support": ["IR:AF_r_5"]},
    {"claim": "Certain humidity regimes can worsen tail latency.", "support": ["LIT_3"]}
  ]
}
```

### Prescriptive
```json
{
  "task": "prescriptive",
  "sample_id": "s12",
  "recommendations": [
    {
      "action": "Increase monitoring cadence and migrate hot data off the drive.",
      "priority": "high",
      "justification": "Instability + error indicators increase failure risk.",
      "support": ["IR:AF_r_187", "LIT_2"]
    }
  ]
}
```

### What‑If
```json
{
  "task": "whatif",
  "sample_id": "s12",
  "counterfactual_statements": [
    {
      "statement": "Reducing RH should decrease tail-latency risk under high-humidity conditions.",
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

## Evaluation metrics (Stage II)

Stage II computes metrics per task and summarizes results.

### Predictive
If the CSV includes `failure` or `label`:
- **Precision, Recall, Accuracy**
- TP/FP/FN/TN counts

Optional regression metrics (if GT columns exist and are predicted):
- `TTF_MSE` (requires `ttf_days`)
- `TL_MSE` (requires `tail_latency_ms`)

### Descriptive & Prescriptive
- **B4 (BLEU‑4)**: requires reference columns (`ref_descriptive`, `ref_prescriptive`)
- **RL (ROUGE‑L)**: same requirement
- **FiP (Faithfulness‑in‑Prompting)**: supported atomic claims / total atomic claims  
  (Support is validated via reference IDs like `IR:*`, `LIT_*`, `ENV:*`.)

### What‑If
- **B4 / RL** if `ref_whatif` exists
- **CFV (Counterfactual Validity)**: checks evidence-linked directional consistency of counterfactual statements.

---

## Troubleshooting

### Stage I: “OPENAI_API_KEY is not set”
```bash
export OPENAI_API_KEY="sk-..."
```

### Stage I: “PyPDF2 not installed”
```bash
pip install PyPDF2
```

### Stage II: BLEU/ROUGE are zero
Your dataset likely lacks `ref_*` columns. That is expected if you do not provide gold references.

---

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{AkewarEtAl_IPDPS_2026,
  author    = {Akewar, Mayur and Madireddy, Sandeep and Luo, Dongsheng and Bhimani, Janki},
  title     = {KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis},
  booktitle = {IEEE International Parallel \\& Distributed Processing Symposium (IPDPS)},
  year      = {2026},
  note      = {To appear}
}
```
