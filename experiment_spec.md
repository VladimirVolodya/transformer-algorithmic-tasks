# Project Specification: Architectural Alignment for Robust Length Generalization

## ROLE AND OBJECTIVE
We test how the allocation of parameters between Attention (routing) and MLP
(logic) affects Length Generalization (OOD robustness) on algorithmic tasks.
Total parameter budget is fixed at ~800k parameters.

---

## PHASE 1: REPOSITORY AUDIT — DONE
Existing codebase already had a decoder-only Transformer, on-the-fly task
generators (ADD / SORT / DYCK / INDEX), a PyTorch train loop in
`src/benchmark.py`, and JSON/CSV logging. Adapted in place (no full rewrite).

---

## PHASE 2: EXPERIMENT SETUP (current)

### 1. Architectures (Decoder-only, 4 layers, ~800k)
| Arch | Role | d_model | n_heads | d_mlp (`d_ff`) |
| --- | --- | ---: | ---: | ---: |
| A | Attention-heavy | 192 | 6 | 192 |
| B | Baseline | 128 | 4 | 512 |
| C | Bottleneck | 80 | 2 | 1088 |
| D | Extreme bottleneck | 48 | 1 | 2048 |

**Position encoding: NoPE** (`pe_variant="nope"`) for all arches.  
`max_len=128`. Causal mask provides order for relative algorithms like addition.

Abandoned alternatives:
* **Absolute PE** — blocked all OOD (zeros on 20–30)
* **RoPE** — ID OK with early stop, but OOD still ~0 on 20–30 (OOD rotation angles)

### 2. Tasks (formats kept from the repo)
Shared 19-token vocabulary. Sequences: `prompt = answer <eos>`.
* **ADDITION:** reversed-digit `rev(a)+rev(b)=rev(sum)`
* **SORT:** `372=237`
* **INDEX:** `372,1=7`
* **DYCK:** Dyck-2 membership → `0`/`1`

### 3. Training & evaluation
* **ID (train / eval):** lengths **1–10**
* **OOD (eval):** lengths **11–15** (soft extrapolation; 20–30 was too hard)
* **Optimizer:** AdamW + OneCycleLR
* **Budget:** up to **15 000** steps per run
* **Early stopping:** every 1000 steps, probe ID exact-match; stop if **ID ≥ 0.99**
* Final metrics: full ID / OOD exact-match means (n_eval=300 per length)

### 4. Grid & logging
* Full grid: **4 Arch × 4 Tasks × 10 Seeds = 160** (after pilots pass)
* **No weight checkpoints**
* Append-only CSV:
  `Task, Architecture, PE, Seed, ID_Accuracy, OOD_Accuracy`
* Entry point: `run_grid.py` (resume-safe; PE is a grid factor)

### 5. Pilot log
| Pilot | Setup | Result |
| --- | --- | --- |
| Absolute 5k | abs, OOD 20–30 | ID ~0.76–0.79; OOD = 0 |
| RoPE #1 | rope, OOD 20–30 | B/D ID ≈ 0.996; OOD = 0 |
| NoPE #2 | nope, OOD 11–15 | B/D ID ≈ 0.99; OOD > 0 |
| ADD 5 seeds | A/B/C/D nope, OOD 11–15 | A best OOD; D unstable |
| DYCK 5 seeds | A/B/C nope, OOD 11–15 | all saturated ~1.0 |
| **Factor ~7h** | **A/B/C × 3 PE × 3 seeds**, ID 1–12, OOD 13–20 | running |

```bash
uv run python run_grid.py --device mps --arches A,B,C --tasks addition \
  --pe-variants nope,rope,absolute --seeds 0,1,2 \
  --train-max-len 12 --ood-min-len 13 --ood-max-len 20 \
  --csv results_add_pe_arch_3seeds.csv
```
