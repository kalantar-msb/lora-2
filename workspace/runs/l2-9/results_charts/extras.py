# Cross-cell summary for l2-9, plus per-cell server-distribution and per-adapter TTFT charts.
# Reuses cells()/load_cell()/TAG/CUT from analysis.py.
import os, numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"; OUT = f"{ROOT}/workspace/runs/l2-9/results_charts"
ns = {}; exec(open(f"{OUT}/analysis.py").read().split("C = {a: cells(a) for a in arms}")[0], ns)
arms, cells, load_cell, pct, COL, TAG, CUT = (ns[k] for k in ("arms", "cells", "load_cell", "pct", "COL", "TAG", "CUT"))

def loads(t, slots, cpu):
    """Two-tier LRU replay of first-token..last-token intervals, as in analysis.py: (slot loads, disk reads)."""
    tot = disk = 0
    for _, g in t[t.adapter != ns["BASE_MODEL"]].groupby("pod"):
        ev = sorted([(r.first_chunk_time_us, 1, r.adapter) for r in g.itertuples()] + [(r.last_chunk_time_us, -1, r.adapter) for r in g.itertuples()], key=lambda x: (x[0], x[1]))
        order = []
        for _, k, ad in ev:
            if k != 1: continue
            if ad in order:
                i = order.index(ad)
                if i >= slots: tot += 1; disk += i >= cpu
                order.pop(i)
            else: tot += 1; disk += 1
            order.insert(0, ad)
    return tot, disk

wls = sorted(set.intersection(*[set(cells(a)) for a in arms]), key=lambda w: (TAG.get(w).rsplit("-", 2)[0], int(TAG.get(w).rsplit("-", 2)[2][:-3]), float(TAG.get(w).rsplit("-", 2)[1][:-3].replace("_", "."))))
rows = []
for wl in wls:
    spec = yaml.safe_load(open(f"{ROOT}/workloads/{wl}.yaml")); order = [c.get("adapter", ns["BASE_MODEL"]) for c in spec["clients"]]
    D = {a: load_cell(cells(a)[wl][0]) for a in arms}; ok = {}
    for a in arms:
        t = D[a]["tr"]; t = t[t.status == "ok"].copy(); t["t_s"] = (t.send_time_us - t.send_time_us.min()) / 1e6
        if wl in CUT: t = t[t.t_s < CUT[wl]]
        ok[a] = t
    tab = {a: ok[a].groupby("adapter").ttft_ms.agg(n="count", p50=lambda s: pct(s, 50), p90=lambda s: pct(s, 90), p99=lambda s: pct(s, 99)).reindex(order) for a in arms}
    for a in arms:
        t, met = ok[a], D[a]["met"]; share = t.pod.value_counts(normalize=True)
        top = t[t.adapter == order[0]].pod.value_counts(normalize=True)
        nl, nd = loads(t, D[a]["slots"], max(D[a]["cpu"], D[a]["slots"])); dur = (t.last_chunk_time_us.max() - t.send_time_us.min()) / 1e6
        rows.append(dict(cell=TAG.get(wl), arm=a, n=len(t), ttft_p50=t.ttft_ms.median(), ttft_p90=pct(t.ttft_ms, 90), ttft_p99=pct(t.ttft_ms, 99),
                         fast=100 * (t.ttft_ms < 1000).mean(), tpot_p50=t.tpot_ms.median(), ttft_mean=t.ttft_ms.mean(), decode_mean=(t.e2e_ms - t.ttft_ms).mean(),
                         e2e_mean=t.e2e_ms.mean(), e2e_p90=pct(t.e2e_ms, 90), e2e_p99=pct(t.e2e_ms, 99),
                         busiest_pct=100 * share.max(), cv=t.pod.value_counts().std(ddof=0) / t.pod.value_counts().mean(),
                         top_adapter_top_pod_pct=100 * top.max(), top_adapter_pods=int((top > 0.01).sum()),
                         loads_per_min=60 * nl / dur, disk_per_min=60 * nd / dur, deferred_max=met.deferred.max() if "deferred" in met else np.nan,
                         running_max=met.running.max() if "running" in met else np.nan,
                         p99_lower=int((tab["loraresidency"]["p99"] < tab["baseline"]["p99"]).sum()) if a == "loraresidency" else np.nan))
    # per-cell charts: server distribution (+TPOT vs load) and per-adapter TTFT p50/p90/p99
    w = 0.8 / len(arms); fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, a in enumerate(arms):
        t, met = ok[a], D[a]["met"]; g = t.groupby("pod")
        df = pd.DataFrame({"share": 100 * g.size() / len(t), "ttft_p99": g.ttft_ms.quantile(.99), "tpot_p50": g.tpot_ms.median(),
                           "running_mean": met.groupby("pod").running.mean() if "running" in met else np.nan}).sort_values("share", ascending=False)
        xs = np.arange(len(df))
        for ax, col, lab in zip(axes[:2], ("share", "ttft_p99"), ("% of requests", "TTFT p99 (ms, log)")):
            bars = ax.bar(xs + i * w, df[col].values, w, color=COL.get(a), label=a)
            for b, p in zip(bars, df.index): ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(p), ha="center", va="bottom", fontsize=6, rotation=90)
            ax.set_ylabel(lab); ax.set_xticks([]); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
        axes[2].scatter(df.running_mean, df.tpot_p50, s=60, color=COL.get(a), label=a)
    axes[0].axhline(25, color="k", ls=":", lw=1); axes[1].set_yscale("log")
    axes[2].set_xlabel("mean requests running (15 s scrapes)"); axes[2].set_ylabel("TPOT p50 (ms)"); axes[2].grid(alpha=.3); axes[2].legend(fontsize=8)
    axes[0].set_title(f"{TAG.get(wl)}: request share per server"); axes[1].set_title("TTFT p99 per server"); axes[2].set_title("TPOT vs load, per server")
    fig.tight_layout(); fig.savefig(f"{OUT}/{TAG.get(wl)}_request_distribution_between_servers.png", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True); xs = np.arange(len(order))
    for ax, q in zip(axes, ("p50", "p90", "p99")):
        for i, a in enumerate(arms): ax.bar(xs + i * w, tab[a][q] / 1e3, w, color=COL.get(a), label=f"{a} (all requests {q}: {pct(ok[a].ttft_ms, int(q[1:]))/1e3:.3g} s)")
        ax.set_yscale("log"); ax.set_ylabel(f"TTFT {q} (s, log)"); ax.grid(axis="y", alpha=.3, which="both"); ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xticks(xs + w / 2); axes[-1].set_xticklabels([f"{x} ({int(tab[arms[0]].loc[x, 'n'])})" for x in order], rotation=60, ha="right", fontsize=7)
    axes[0].set_title(f"{TAG.get(wl)}: per-adapter TTFT p50 / p90 / p99 — adapters by popularity (request count in brackets)")
    fig.tight_layout(); fig.savefig(f"{OUT}/{TAG.get(wl)}_per_adapter_ttft.png", dpi=120); plt.close(fig)

S = pd.DataFrame(rows); pd.set_option("display.width", 250)
S.to_csv(f"{OUT}/summary.csv", index=False)
print(S.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
# summary chart: loraresidency relative to baseline, per cell
P = S.pivot(index="cell", columns="arm").reindex(S.cell.unique())
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (m, lab) in zip(axes, (("ttft_p90", "TTFT p90 (s)"), ("ttft_p99", "TTFT p99 (s)"), ("e2e_mean", "E2E mean (s)"))):
    xs = np.arange(len(P))
    for i, a in enumerate(arms): ax.bar(xs + i * 0.4, P[(m, a)] / 1e3, 0.4, color=COL.get(a), label=a)
    ax.set_xticks(xs + 0.2); ax.set_xticklabels(P.index, rotation=45, ha="right", fontsize=8); ax.set_yscale("log"); ax.set_ylabel(lab + ", log"); ax.grid(axis="y", alpha=.3, which="both"); ax.legend(fontsize=8)
axes[0].set_title("l2-9: all cells (a = Zipf alpha, rps = req/s, tok = mean output tokens)")
fig.tight_layout(); fig.savefig(f"{OUT}/summary_all_cells.png", dpi=120); print("Saved:", f"{OUT}/summary_all_cells.png")
