# Figures and numbers for the two-page summary. Latency comes from the client traces only
# (adapter requests unless noted); slot-queue and disk-read curves come from load_trend.csv.
import base64, io, json
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"
RES = f"{ROOT}/workspace/runs/l2-9/results"; CH = f"{ROOT}/workspace/runs/l2-9/results_charts"; OUT = f"{ROOT}/workspace/runs/l2-9/report"
BM = "ibm-granite/granite-3.3-2b-instruct"
COL = {"baseline": "#3567a6", "loraresidency": "#d9711c"}; LAB = {"baseline": "baseline (lora-affinity)", "loraresidency": "loraresidency"}
INK, GRID = "#2b2f36", "#d9dce1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#9aa0a8", "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6})
pct = lambda s, q: float(np.percentile(s, q))


def trace(arm, wl):
    t = pd.read_csv(f"{RES}/{arm}/{wl}/i1/trace_data.csv"); t = t[t.status == "ok"].copy()
    t["adapter"] = t.adapter.fillna(BM)
    t["ttft"] = (t.first_chunk_time_us - t.send_time_us) / 1e6; t["e2e"] = (t.last_chunk_time_us - t.send_time_us) / 1e6
    t["tpot"] = np.where(t.output_tokens > 1, (t.last_chunk_time_us - t.first_chunk_time_us) / (t.output_tokens - 1) / 1e3, np.nan)
    return t


CELLS = [  # (group, label, workload)
    ("Load", "7.5 req/s", "granite-1_5-27adapter.workload-7_5-32"), ("Load", "15 req/s", "granite-1_5-27adapter.workload-15-32"),
    ("Load", "22.5 req/s", "granite-1_5-27adapter.workload-22_5-32"), ("Load", "30 req/s", "granite-1_5-27adapter.workload-30-32"),
    ("Load", "45 req/s", "granite-1_5-27adapter.workload-45-32"),
    ("Output length 128 tokens", "7.5 req/s", "granite-1_5-27adapter.workload-7_5-128"), ("Output length 128 tokens", "15 req/s", "granite-1_5-27adapter.workload-15-128"),
    ("Popularity α = 1.0", "15 req/s", "granite-1_0-27adapter.workload-15-32"), ("Popularity α = 1.0", "30 req/s", "granite-1_0-27adapter.workload-30-32"),
    ("Plus 20% base-model traffic", "15 req/s", "granite-1_5-27a-base20.workload-15-32"), ("Plus 20% base-model traffic", "30 req/s", "granite-1_5-27a-base20.workload-30-32"),
    ("Plus 20% base-model traffic", "15 req/s, 128 tok", "granite-1_5-27a-base20.workload-15-128"),
]
rows = []
for grp, lab, wl in CELLS:
    order = [c["adapter"] for c in yaml.safe_load(open(f"{ROOT}/workloads/{wl}.yaml"))["clients"] if "adapter" in c]
    T = {a: trace(a, wl) for a in COL}; A = {a: T[a][T[a].adapter != BM] for a in COL}
    p99 = {a: A[a].groupby("adapter").ttft.quantile(.99).reindex(order) for a in COL}
    r = dict(group=grp, label=lab, wl=wl, lower=int((p99["loraresidency"] < p99["baseline"]).sum()), n_adapters=len(order),
             minutes=(T["baseline"].send_time_us.max() - T["baseline"].send_time_us.min()) / 6e7)
    for a in COL:
        s = A[a]; r |= {f"{a}_p90": pct(s.ttft, 90), f"{a}_p99": pct(s.ttft, 99), f"{a}_e2e": s.e2e.mean(), f"{a}_tpot": s.tpot.median()}
    rows.append(r)
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/cells.csv", index=False)
pd.set_option("display.width", 250)
print(C[["group", "label", "minutes", "baseline_p90", "loraresidency_p90", "baseline_p99", "loraresidency_p99", "baseline_e2e",
         "loraresidency_e2e", "baseline_tpot", "loraresidency_tpot", "lower"]].round(2).to_string(index=False))

L = pd.read_csv(f"{CH}/load_trend.csv"); xs = sorted(L.rps.unique())
figs = {}


def emit(fig, name):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    open(f"{OUT}/{name}.png", "wb").write(buf.getvalue()); figs[name] = base64.b64encode(buf.getvalue()).decode()


def loadaxis(ax):
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.set_xlabel("load (adapter requests per second)")


def line(ax, col, a, df=L, **kw):
    s = df[df.arm == a]; ax.plot(s.rps, s[col], marker="o", ms=4.5, lw=2, color=COL[a], label=LAB[a], **kw)


def logsec(ax, ticks=(0.2, 0.5, 1, 2, 5, 10, 20, 50)):
    ax.set_yscale("log"); ax.set_yticks(ticks); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g} s"))


# Figure 1: TTFT p90 and p99 against load
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
for ax, q, t in zip(axes, ("p90", "p99"), ("TTFT p90", "TTFT p99")):
    for a in COL: line(ax, q, a)
    logsec(ax); loadaxis(ax); ax.set_title(t, loc="left")
    ax.axvspan(15, 22.5, color="#eef0f3", zorder=0, lw=0)
axes[1].text(18.75, 1.6, "adapter slots\nrun out", ha="center", va="center", fontsize=7.5, color="#5c6370")
axes[0].legend(loc="upper left", fontsize=8)
emit(fig, "fig1")

# Figure 2: TTFT p90 by popularity group
groups = [("rank 1", "rank 1 (45% of traffic)"), ("ranks 2-3", "ranks 2–3 (25%)"), ("ranks 4-9", "ranks 4–9 (19%)"), ("ranks 10-27", "ranks 10–27 (12%)")]
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5), sharey=True)
for ax, (g, t) in zip(axes, groups):
    for a in COL: line(ax, f"{g} p90", a)
    logsec(ax, (0.2, 1, 5, 20)); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7.5); ax.set_title(t, loc="left", fontsize=8.5)
axes[0].set_ylabel("TTFT p90"); fig.supxlabel("load (adapter requests per second)", fontsize=9, y=-0.04, color=INK)
axes[0].legend(loc="upper left", fontsize=7.5)
emit(fig, "fig2")

# Figure 3: mechanism. Disk reads omit baseline 7.5 (router logs cover 5% of it); loraresidency 22.5 is hollow (80%).
fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.7))
for a in COL: line(axes[0], "deferred_max", a)
axes[0].set_title("Most requests waiting for an adapter slot, per pod", loc="left"); loadaxis(axes[0]); axes[0].legend(loc="upper left", fontsize=8)
for a in COL:
    s = L[(L.arm == a) & ~((a == "baseline") & (L.rps == 7.5))]
    axes[1].plot(s.rps, s.disk_per_req, marker="o", ms=4.5, lw=2, color=COL[a], label=LAB[a])
    if a == "loraresidency":
        p = s[s.rps == 22.5]; axes[1].plot(p.rps, p.disk_per_req, "o", ms=6, mfc="white", mec=COL[a], mew=1.6)
axes[1].set_title("Adapter reads from disk per request (estimated)", loc="left"); loadaxis(axes[1])
emit(fig, "fig3")

# Figure 4: the other dimensions, as improvement over baseline
Ci = C.set_index("wl")
pairs = [("32 tokens", ("granite-1_5-27adapter.workload-7_5-32", "granite-1_5-27adapter.workload-15-32"),
          "128 tokens", ("granite-1_5-27adapter.workload-7_5-128", "granite-1_5-27adapter.workload-15-128"), ("7.5", "15")),
         ("α = 1.5", ("granite-1_5-27adapter.workload-15-32", "granite-1_5-27adapter.workload-30-32"),
          "α = 1.0", ("granite-1_0-27adapter.workload-15-32", "granite-1_0-27adapter.workload-30-32"), ("15", "30"))]
metrics = [("p90", lambda w: Ci.loc[w, "baseline_p90"] / Ci.loc[w, "loraresidency_p90"], "TTFT p90, times lower", lambda v: f"{v:.1f}×"),
           ("e2e", lambda w: 100 * (1 - Ci.loc[w, "loraresidency_e2e"] / Ci.loc[w, "baseline_e2e"]), "E2E mean, % lower", lambda v: f"{v:.0f}%")]
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6))
for k, (la, wa, lb, wb, rates) in enumerate(pairs):
    for j, (m, f, ttl, fmt) in enumerate(metrics):
        ax = axes[2 * k + j]; x = np.arange(2); w = .36
        va, vb = [f(v) for v in wa], [f(v) for v in wb]
        ba = ax.bar(x - w / 2, va, w, color="#b7c2cf", label=la); bb = ax.bar(x + w / 2, vb, w, color="#4b5563", label=lb)
        for bars, vals in ((ba, va), (bb, vb)):
            for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v, fmt(v), ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([f"{r} req/s" for r in rates], fontsize=8); ax.set_title(ttl, loc="left", fontsize=8.5)
        ax.margins(y=.22); ax.grid(axis="x", visible=False)
        if m == "p90": ax.axhline(1, color="#9aa0a8", lw=.8)
        ax.legend(fontsize=7.5, loc="upper left")
emit(fig, "fig4")
json.dump(figs, open(f"{OUT}/figs.json", "w"))
