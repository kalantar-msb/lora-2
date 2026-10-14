# Latency-against-load curves for the alpha=1.5, 32-token cells (7.5 to 45 req/s), one line per arm.
# Reuses cells()/load_cell() from analysis.py. Dashed lines in the first chart drop each cell's first
# WARM seconds, to show how much start-up adapter loads move each point.
import re, numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"; OUT = f"{ROOT}/workspace/runs/l2-9/results_charts"
ns = {}; exec(open(f"{OUT}/analysis.py").read().split("C = {a: cells(a) for a in arms}")[0], ns)
arms, cells, load_cell, pct, COL = ns["arms"], ns["cells"], ns["load_cell"], ns["pct"], ns["COL"]
WARM = 30.0
GROUPS = [(1, 1, "rank 1"), (2, 3, "ranks 2-3"), (4, 9, "ranks 4-9"), (10, 27, "ranks 10-27")]

def loads(t, slots, cpu):
    """Two-tier LRU replay of first-token..last-token intervals, as in analysis.py: (slot loads, disk reads)."""
    tot = disk = 0
    for _, g in t.groupby("pod"):
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

wls = [w for w in set.intersection(*[set(cells(a)) for a in arms]) if re.match(r"granite-1_5-27adapter\.workload-[0-9_]+-32$", w)]
rate = lambda w: float(re.search(r"workload-([0-9_]+)-32$", w).group(1).replace("_", "."))
wls.sort(key=rate)
rows = []
for wl in wls:
    order = [c["adapter"] for c in yaml.safe_load(open(f"{ROOT}/workloads/{wl}.yaml"))["clients"]]; rank = {a: i + 1 for i, a in enumerate(order)}
    for a in arms:
        D = load_cell(cells(a)[wl][0]); t = D["tr"]; t = t[t.status == "ok"].copy(); met = D["met"]
        t["t_s"] = (t.send_time_us - t.send_time_us.min()) / 1e6; t["rank"] = t.adapter.map(rank); warm = t[t.t_s >= WARM]
        share = t.pod.value_counts(normalize=True); top = t[t.adapter == order[0]].pod.value_counts(normalize=True)
        nl, nd = loads(t, D["slots"], max(D["cpu"], D["slots"]))
        r = dict(rps=rate(wl), arm=a, n=len(t), dur_s=t.t_s.max(),
                 p50=t.ttft_ms.median() / 1e3, p90=pct(t.ttft_ms, 90) / 1e3, p99=pct(t.ttft_ms, 99) / 1e3,
                 w_p50=warm.ttft_ms.median() / 1e3, w_p90=pct(warm.ttft_ms, 90) / 1e3, w_p99=pct(warm.ttft_ms, 99) / 1e3,
                 under1=100 * (t.ttft_ms < 1000).mean(), under2=100 * (t.ttft_ms < 2000).mean(), e2e_mean=t.e2e_ms.mean() / 1e3,
                 busiest=100 * share.max(), top_on_one=100 * top.max(), cv=t.pod.value_counts().std(ddof=0) / t.pod.value_counts().mean(),
                 deferred_max=met.deferred.max(), loads_per_req=nl / len(t), disk_per_req=nd / len(t), tpot_p50=t.tpot_ms.median())
        for lo, hi, lab in GROUPS:
            g = t[(t["rank"] >= lo) & (t["rank"] <= hi)].ttft_ms
            r[f"{lab} p50"] = g.median() / 1e3; r[f"{lab} p90"] = pct(g, 90) / 1e3
        rows.append(r)
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/load_trend.csv", index=False)
pd.set_option("display.width", 250)
print(S[["rps", "arm", "n", "dur_s", "p50", "p90", "p99", "w_p90", "w_p99", "under1", "under2", "e2e_mean", "busiest", "top_on_one", "deferred_max", "loads_per_req", "disk_per_req", "tpot_p50"]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print(S[["rps", "arm"] + [c for c in S.columns if c.startswith("rank")]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

xs = S.rps.unique(); saved = []
def axis(ax, ylab, log=False):
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.set_xlabel("load (req/s)"); ax.set_ylabel(ylab); ax.grid(alpha=.3, which="both")
    if log: ax.set_yscale("log")
def save(fig, name):
    fig.tight_layout(); p = f"{OUT}/{name}"; fig.savefig(p, dpi=120); plt.close(fig); saved.append(p)
def line(ax, col, a, **kw):
    s = S[S.arm == a]; ax.plot(s.rps, s[col], marker="o", color=COL.get(a), **kw)
note = "alpha=1.5, 32-token outputs, 4 pods x 3 GPU slots; one run per point; cells run 40 min (7.5) down to ~1 min (45)"

# 1. TTFT percentiles
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, q in zip(axes, ("p50", "p90", "p99")):
    for a in arms:
        line(ax, q, a, label=f"{a}"); line(ax, f"w_{q}", a, ls="--", alpha=.6, label=f"{a}, first {WARM:.0f} s dropped")
    axis(ax, f"TTFT {q} (s, log)", log=True); ax.set_title(f"TTFT {q}"); ax.legend(fontsize=7)
fig.suptitle(f"TTFT against load — {note}", fontsize=9); save(fig, "load_ttft_percentiles.png")
# 2. Targets met, and E2E
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (col, lab) in zip(axes, (("under1", "% requests with TTFT < 1 s"), ("under2", "% requests with TTFT < 2 s"), ("e2e_mean", "E2E mean (s)"))):
    for a in arms: line(ax, col, a, label=a)
    axis(ax, lab); ax.set_title(lab); ax.legend(fontsize=8)
fig.suptitle(f"Requests meeting a TTFT target, and mean E2E, against load — {note}", fontsize=9); save(fig, "load_targets_and_e2e.png")
# 3. TTFT by popularity group
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
for ax, (_, _, lab) in zip(axes, GROUPS):
    for a in arms:
        line(ax, f"{lab} p90", a, label=f"{a} p90"); line(ax, f"{lab} p50", a, ls="--", alpha=.7, label=f"{a} p50")
    axis(ax, "TTFT (s, log)", log=True); ax.set_title(f"adapters {lab} by popularity"); ax.legend(fontsize=7)
fig.suptitle(f"TTFT against load, by adapter popularity — {note}", fontsize=9); save(fig, "load_ttft_by_popularity.png")
# 4. Concentration
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (col, lab) in zip(axes, (("busiest", "busiest pod's % of requests"), ("top_on_one", "top adapter: % of its requests on one pod"), ("cv", "spread of requests across pods (CV)"))):
    for a in arms: line(ax, col, a, label=a)
    if col == "busiest": ax.axhline(25, color="k", ls=":", lw=1, label="even split")
    axis(ax, lab); ax.set_title(lab); ax.legend(fontsize=8)
fig.suptitle(f"How requests are spread across pods, against load — {note}", fontsize=9); save(fig, "load_concentration.png")
# 5. Bottleneck and cost
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
for ax, (col, lab) in zip(axes, (("deferred_max", "max requests waiting for an adapter slot (per pod)"), ("loads_per_req", "adapter slot loads per request (est.)"),
                                 ("disk_per_req", "adapter disk reads per request (est.)"), ("tpot_p50", "TPOT p50 (ms)"))):
    for a in arms: line(ax, col, a, label=a)
    axis(ax, lab); ax.set_title(lab, fontsize=9); ax.legend(fontsize=8)
fig.suptitle(f"Adapter-slot bottleneck, load cost and TPOT, against load — {note}", fontsize=9); save(fig, "load_bottleneck.png")
for p in saved: print("Saved:", p)
