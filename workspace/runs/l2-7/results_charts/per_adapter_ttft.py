import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"; OUT = f"{ROOT}/workspace/runs/l2-7/results_charts"
ns = {}; exec(open(f"{OUT}/analysis.py").read().split("C = {a: cells(a) for a in arms}")[0], ns)
arms, cells, load_cell, pct, COL = ns["arms"], ns["cells"], ns["load_cell"], ns["pct"], ns["COL"]
wl = "granite-27adapter.workload"
order = [c["adapter"] for c in yaml.safe_load(open(f"{ROOT}/workloads/{wl}.yaml"))["clients"]]
ok = {}
for a in arms:
    t = pd.concat([load_cell(d)["tr"] for d in cells(a)[wl]], ignore_index=True); ok[a] = t[t.status == "ok"]
tab = {a: ok[a].groupby("adapter").ttft_ms.agg(n="count", p50=lambda s: pct(s, 50), p90=lambda s: pct(s, 90), p99=lambda s: pct(s, 99)).reindex(order) for a in arms}
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
xs = np.arange(len(order)); w = 0.8 / len(arms)
for ax, q in zip(axes, ("p50", "p90", "p99")):
    for i, a in enumerate(arms):
        ax.bar(xs + i * w, tab[a][q] / 1e3, w, color=COL.get(a), label=f"{a} (all requests {q}: {pct(ok[a].ttft_ms, int(q[1:]))/1e3:.3g} s)")
    ax.set_yscale("log"); ax.set_ylabel(f"TTFT {q} (s, log)"); ax.grid(axis="y", alpha=.3, which="both"); ax.legend(fontsize=8, loc="upper left")
axes[-1].set_xticks(xs + w * (len(arms) - 1) / 2)
axes[-1].set_xticklabels([f"{x} ({int(tab[arms[0]].loc[x, 'n'])})" for x in order], rotation=60, ha="right", fontsize=7)
axes[0].set_title("Per-adapter TTFT p50 / p90 / p99 — adapters by popularity (request count in brackets)")
fig.tight_layout(); p = f"{OUT}/per_adapter_ttft.png"; fig.savefig(p, dpi=120); print("Saved:", p)
