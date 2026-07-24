# How Transformer Architecture Design Choices Affect Length Generalization on Algorithmic Tasks

*Merge-ready section for the joint paper. Numbers from the controlled ~800k ADD / DYCK runs in this repository. Figure paths are relative to the repo root.*

## Contributions (for coauthors)

1. **Positional encoding is the primary barrier** to length extrapolation on reversed-digit addition: under Absolute PE and RoPE, OOD exact-match is ~0 even on soft gaps; only **NoPE** yields non-zero OOD.
2. Under NoPE, with a fixed ~800k budget, **Attention/MLP allocation** modulates both OOD accuracy and ID optimization speed; OOD is **non-monotonic** in attention share (peak near ~50%, not at maximum attention).
3. **DYCK** at the same soft OOD gap saturates (~1.0) across architectures and does not discriminate inductive biases.

**Suggested figure placement in the joint article**

| Joint label | Content | Asset |
| --- | --- | --- |
| Fig. PE comparison | Absolute / RoPE / NoPE on ADD (ID vs OOD) | narrative + `results_add_pe_arch_3seeds.csv`, soft-OOD PE probe |
| Fig. OOD vs attn | OOD exact-match vs attention share (merged by `n_heads`) | [`figures/ood_vs_attn_frac.png`](../figures/ood_vs_attn_frac.png) |
| Fig. steps vs attn | Steps to ID ≥ 0.99 vs attention share (same x-axis) | [`figures/convergence_vs_attn_frac.png`](../figures/convergence_vs_attn_frac.png) |

---

## Abstract

This study investigates the fragility of length generalization in Transformers on algorithmic tasks (Addition, DYCK). We examine two design choices—**positional encoding (PE)** and the **parameter allocation between Attention (routing) and MLP (logic)**—under a fixed budget of ~0.8M parameters. Models are trained on short sequences (lengths 1–10) and evaluated on longer sequences (soft OOD 11–15; far OOD 13–20 / 20–30). For addition, PE is the primary bottleneck: only **NoPE** yields non-zero OOD exact-match. Once PE is removed, performance depends on routing capacity: attention-heavier models learn ID faster and generalize better than extreme MLP bottlenecks, but OOD peaks at intermediate attention share rather than increasing monotonically.

---

## 1. Experimental Setup

### Parameter budget and architectures

All models are **decoder-only Transformers with 4 layers** and a shared 19-token task vocabulary. We hold total parameters near **~800k** and vary \(d_{\mathrm{model}}\), \(n_{\mathrm{heads}}\), and \(d_{\mathrm{ff}}\) (\(d_{\mathrm{mlp}}\)) to move along an Attention↔MLP continuum. Attention share is

\[
\mathrm{attn\_frac} = \frac{4 L d_{\mathrm{model}}^2}{4 L d_{\mathrm{model}}^2 + 2 L d_{\mathrm{model}} d_{\mathrm{ff}}}.
\]

Named checkpoints used in runs (NoPE by default for the architecture sweep):

| Arch | Role | \(d_{\mathrm{model}}\) | \(H\) | \(d_{\mathrm{ff}}\) | attn_frac (approx.) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Attention-heavy | 192 | 6 | 192 | 0.67 |
| E | Mid attention | 160 | 5 | 320 | 0.50 |
| B | Baseline (4× MLP) | 128 | 4 | 512 | 0.33 |
| F | Mild bottleneck | 112 | 4 | 672 | 0.25 |
| C | Bottleneck | 80 | 2 | 1088 | 0.13 |
| G | Strong bottleneck | 64 | 2 | 1408 | 0.08 |
| D | Extreme bottleneck | 48 | 1 | 2048 | 0.05 |

Additional points on the same continuum (H: 7 heads / attn≈0.75; I: 3 heads / attn≈0.20) densify the sweep. **For the two main plots, runs are pooled by \(n_{\mathrm{heads}}\)**; the x-position is the mean attention share within each head-count group (so B+F share the 4-head point, C+G the 2-head point). Plot markers are unlabeled (no arch letters / head counts on the figure).

### Tasks, lengths, metric, optimization

- **ADDITION:** reversed-digit format `rev(a)+rev(b)=rev(sum)`; on-the-fly sampling.
- **DYCK:** Dyck-2 membership → `0`/`1`.
- **ID / train lengths:** 1–10. **Soft OOD:** 11–15 (primary architecture comparison). **Far OOD:** 13–20 and 20–30 used in PE probes (typically collapse to 0 under Absolute/RoPE).
- **Metric:** greedy autoregressive **full-sequence exact match** (mean over length buckets; \(n_{\mathrm{eval}}=300\) per length).
- **Optimization:** AdamW + OneCycleLR; up to **15 000** steps; **early stop** when ID exact-match ≥ **0.99** (probe every 1 000 steps). Batch size 256, lr \(3\times10^{-4}\).

---

## 2. Key Findings

### Finding 1: Positional encoding dictates OOD survival

Architecture changes alone do not overcome failures induced by explicit PE on length extrapolation for addition.

- **Absolute PE:** OOD exact-match ≈ **0** on far and soft extrapolations (unseen absolute positions).
- **RoPE:** OOD ≈ **0** on far jumps (20–30) and on mid jumps (13–20). On soft OOD 11–15, RoPE/Absolute remain ≈ **0** while NoPE is already non-zero—so the PE failure is not only an artifact of aggressive length gaps.
- **NoPE:** **Only** variant with non-zero length generalization on ADD (typically ~0.14–0.32 mean OOD on 11–15 depending on architecture). Order is carried by the causal mask; the model avoids binding to absolute length indices.

**Bridge.** PE decides *whether* OOD is possible; under NoPE, the Attention/MLP ratio modulates *how much* OOD accuracy is achieved and *how fast* ID reaches the early-stop threshold.

### Finding 2: Routing-heavy bias under NoPE (OOD vs attention share)

![OOD exact-match vs attention share (ADD, NoPE, ID 1–10, OOD 11–15; pooled by n_heads)](../figures/ood_vs_attn_frac.png)

Under NoPE we expected addition to favor MLP-heavy “logic” bottlenecks. Empirically the opposite holds for routing: without PE, heads must resolve relative digit alignment via the causal mask.

**Summary (pooled by \(n_{\mathrm{heads}}\), soft OOD 11–15):**

| Heads | attn share (group mean) | Mean OOD | Notes |
| ---: | ---: | ---: | --- |
| 7 | ~0.75 | ~0.20 | High attention, below mid-peak |
| 6 | ~0.67 | ~0.30 | Strong |
| 5 | ~0.50 | **~0.32** | **Peak** |
| 4 | ~0.30 | ~0.22 | B+F pooled |
| 3 | ~0.20 | ~0.19 | |
| 2 | ~0.11 | ~0.22 | C+G pooled; high variance |
| 1 | ~0.05 | ~0.06 | Collapse / unstable seeds |

OOD is **non-monotonic** in attention share: the best mean is near **~50%** attention (5 heads), not at maximum attention (7 heads / ~75%). Extreme 1-head bottlenecks underperform and show failed ID runs on some seeds. This supports architectural alignment with a **routing-heavy** profile once PE is removed, without claiming “more attention is always better.”

### Finding 3: Optimization dynamics and DYCK saturation

![Steps to ID ≥ 0.99 vs attention share (ADD, NoPE; pooled by n_heads)](../figures/convergence_vs_attn_frac.png)

**ID convergence (steps until early-stop ID ≥ 0.99), NoPE, lengths 1–10** (pooled by heads; same x-axis as Fig. OOD):

| Heads | attn≈ | Mean steps (±std) | \(n\) |
| ---: | ---: | ---: | ---: |
| 7 | 0.75 | ~6800 ± 1100 | 5 |
| 6 | 0.67 | ~6800 ± 450 | 5 |
| 5 | 0.50 | ~7500 ± 840 | 6 |
| 4 | 0.29 | ~8000 ± 630 | 6 |
| 3 | 0.20 | ~8600 ± 1140 | 5 |
| 2 | 0.11 | ~10300 ± 820 | 6 |
| 1 | ~0.05 | *(fill-in runs; see CSV when complete)* | — |

Higher attention share is associated with **faster** ID convergence. Severe attention bottlenecks make hitting ID ≥ 0.99 harder (more steps; unstable ID on extreme 1-head settings in the OOD grid). This is consistent with Finding 2 but measures trainability, not OOD directly.

**DYCK saturation.** On soft OOD 11–15, architectures A/B/C all reach **~1.0** OOD exact-match (5 seeds). The gap is too easy to discriminate Attention vs MLP bias; DYCK is reported as a saturated control, not as evidence for architecture ranking.

---

## 3. Discussion and Limitations

Length generalization on these tasks is constrained by inductive bias, not only by scale. PE acts as a hard gate for addition OOD; under NoPE, routing capacity shapes both generalization and optimization, with a mid-range attention optimum for OOD.

**Limitations**

1. **Exact-match penalty.** One wrong digit zeros the example; underlying OOD competence may exceed the reported ~0.30 peak.
2. **Seed imbalance.** Some head groups have fewer seeds (e.g. 4-head mild / 2-head strong points with \(n=3\)); variance bands are large.
3. **Soft OOD.** Primary architecture claims use 11–15; far OOD remains near zero for Absolute/RoPE and is weak even for NoPE.
4. **Optimizer.** Unstable extreme bottlenecks may need stronger optimization (e.g. modern second-order / Muon-style methods)—left to future work.
5. **Formats.** Scratchpad / chain-of-thought formats are a natural next step to raise OOD toward 1.0 while keeping the same PE and Attention/MLP controls.

---

## Conclusion

Length generalization on algorithmic addition is gated first by **positional encoding**: Absolute and RoPE block OOD exact-match in our setting; **NoPE** is necessary for non-zero extrapolation. Conditional on NoPE, a fixed-budget Attention↔MLP sweep shows that **routing capacity** improves ID convergence and OOD accuracy relative to extreme MLP bottlenecks, with a **non-monotonic** OOD peak near intermediate attention share. Aligning PE, architecture ratio, and task format is therefore essential; scale alone does not explain the failure modes we observe.

---

## Appendix: Data and figures for merge

| Artifact | Path |
| --- | --- |
| OOD vs attn plot | [`figures/ood_vs_attn_frac.png`](../figures/ood_vs_attn_frac.png) |
| Convergence vs attn plot | [`figures/convergence_vs_attn_frac.png`](../figures/convergence_vs_attn_frac.png) |
| OOD summary by arch | [`results_final_ood.csv`](../results_final_ood.csv) |
| Per-run OOD table | [`results_final_ood_runs.csv`](../results_final_ood_runs.csv) |
| Steps to ID ≥ 0.99 | [`results_convergence_steps.csv`](../results_convergence_steps.csv) |
| ADD NoPE 5-seed base (A–D) | [`results_add_5seeds.csv`](../results_add_5seeds.csv) |
| Extra arches / E top-up | [`results_add_attn_ratio.csv`](../results_add_attn_ratio.csv), [`results_add_heads_extra.csv`](../results_add_heads_extra.csv) |
| PE × arch factor (ID 1–12, OOD 13–20) | [`results_add_pe_arch_3seeds.csv`](../results_add_pe_arch_3seeds.csv) |
| DYCK 5 seeds | [`results_dyck_5seeds.csv`](../results_dyck_5seeds.csv) |
| Rebuild OOD summary | `uv run python scripts/build_final_ood_table.py` |
| Rebuild convergence plot | `uv run python scripts/plot_convergence_vs_attn.py` |
