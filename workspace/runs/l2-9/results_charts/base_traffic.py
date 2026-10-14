# Base-model traffic: latency of the 20% base-model requests per arm, whether adding them changed the
# adapter requests (against the same cell without base traffic), and how each arm spread them over pods.
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"; OUT = f"{ROOT}/workspace/runs/l2-9/results_charts"
ns = {}; exec(open(f"{OUT}/analysis.py").read().split("C = {a: cells(a) for a in arms}")[0], ns)
arms, cells, load_cell, pct, COL, TAG, BM = (ns[k] for k in ("arms", "cells", "load_cell", "pct", "COL", "TAG", "BASE_MODEL"))
PAIRS = [(f"granite-1_5-27a-base20.workload-{c}", f"granite-1_5-27adapter.workload-{c}") for c in ("15-32", "30-32", "15-128")]
def stats(t):
    return dict(n=len(t), ttft_p50=t.ttft_ms.median(), ttft_p90=pct(t.ttft_ms, 90), ttft_p99=pct(t.ttft_ms, 99), under1=100 * (t.ttft_ms < 1000).mean(),
                tpot_p50=t.tpot_ms.median(), tpot_p90=t.tpot_ms.quantile(.9), e2e_mean=t.e2e_ms.mean(), e2e_p99=pct(t.e2e_ms, 99))
rows, pods = [], []
for wb, w0 in PAIRS:
    for a in arms:
        d = load_cell(cells(a)[wb][0]); t = d["tr"][d["tr"].status == "ok"]; met = d["met"]
        t0 = load_cell(cells(a)[w0][0])["tr"]; t0 = t0[t0.status == "ok"]
        base, lora = t[t.adapter == BM], t[t.adapter != BM]
        rows += [dict(cell=TAG.get(wb), arm=a, requests="base model", **stats(base)),
                 dict(cell=TAG.get(wb), arm=a, requests="adapters (with base traffic)", **stats(lora)),
                 dict(cell=TAG.get(wb), arm=a, requests="adapters (same cell, no base traffic)", **stats(t0))]
        g = t.groupby("pod")
        for p in t.pod.dropna().unique():
            b = base[base.pod == p]
            pods.append(dict(cell=TAG.get(wb), arm=a, pod=p, base_share=100 * len(b) / len(base), adapter_share=100 * (lora.pod == p).mean(),
                             adapters_served=lora[lora.pod == p].adapter.nunique(), running_mean=met[met.pod == p].running.mean(),
                             base_ttft_p99=pct(b.ttft_ms, 99), base_tpot_p50=b.tpot_ms.median()))
S = pd.DataFrame(rows); P = pd.DataFrame(pods).sort_values(["cell", "arm", "adapter_share"], ascending=[True, True, False])
S.to_csv(f"{OUT}/base_traffic_summary.csv", index=False); P.to_csv(f"{OUT}/base_traffic_by_pod.csv", index=False)
pd.set_option("display.width", 250); f = lambda x: f"{x:,.1f}"
print("=== Latency (ms) ===\n" + S.to_string(index=False, float_format=f))
print("\n=== Base-model requests by pod (shares in %) ===\n" + P.to_string(index=False, float_format=f))
# chart: base-model latency per arm, and adapter TTFT p90 with/without base traffic
cellnames = S.cell.unique(); xs = np.arange(len(cellnames)); w = 0.38
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
for ax, (m, lab, log) in zip(axes[:3], (("ttft_p50", "base-model TTFT p50 (ms)", False), ("ttft_p99", "base-model TTFT p99 (ms, log)", True), ("tpot_p50", "base-model TPOT p50 (ms)", False))):
    for i, a in enumerate(arms):
        v = [S[(S.cell == c) & (S.arm == a) & (S.requests == "base model")][m].iloc[0] for c in cellnames]
        bars = ax.bar(xs + i * w, v, w, color=COL.get(a), label=a)
        for b, y in zip(bars, v): ax.text(b.get_x() + b.get_width() / 2, y, f"{y:,.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs + w / 2); ax.set_xticklabels(cellnames, rotation=20, ha="right", fontsize=8); ax.set_title(lab, fontsize=9); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    if log: ax.set_yscale("log")
ax = axes[3]; ww = 0.2
for i, a in enumerate(arms):
    for j, (req, hatch) in enumerate((("adapters (same cell, no base traffic)", ""), ("adapters (with base traffic)", "//"))):
        v = [S[(S.cell == c) & (S.arm == a) & (S.requests == req)].ttft_p90.iloc[0] / 1e3 for c in cellnames]
        ax.bar(xs + (2 * i + j) * ww, v, ww, color=COL.get(a), hatch=hatch, edgecolor="k", lw=.3, label=f"{a}, {'with' if j else 'without'} base traffic")
ax.set_yscale("log"); ax.set_xticks(xs + 1.5 * ww); ax.set_xticklabels(cellnames, rotation=20, ha="right", fontsize=8); ax.set_title("adapter-request TTFT p90 (s, log)", fontsize=9); ax.grid(axis="y", alpha=.3, which="both"); ax.legend(fontsize=7)
fig.suptitle("Base-model traffic (20% of requests, added on top of the adapter traffic)", fontsize=10)
fig.tight_layout(); p = f"{OUT}/base_traffic.png"; fig.savefig(p, dpi=120); plt.close(fig); print("\nSaved:", p)
