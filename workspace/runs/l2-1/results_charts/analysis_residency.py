"""Label each request hot / warm / cold-cpu / cold-disk by reconstructing per-pod
adapter residency from the trace + EPP routing log, then compare TTFT by label.

Residency model (vLLM, max_loras=3 GPU slots, max_cpu_loras=6 CPU slots, LRU):
  at request R's send time t on pod P, rank adapters by last use on P
  (in use now = most recent; else latest last_chunk <= t; warm-up EPP decisions
  count as point uses).
  hot       another request for R's adapter is executing on P at t
  warm      not executing, but within the 3 most recent       -> still in a GPU slot
  cold-cpu  outside the top 3, within the top 6               -> in CPU, must swap in
  cold-disk outside the top 6, or never used on P             -> must load from disk
"""
import glob, json, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = "l2-1"
base = f"workspace/runs/{run}/results"
out = f"workspace/runs/{run}/results_charts"
WL = "granite-9adapter.workload"
GPU_SLOTS, CPU_SLOTS = 3, 6
arms = ["baseline"] + sorted(a for a in os.listdir(base) if a != "baseline" and os.path.isdir(f"{base}/{a}"))
LABELS = ["hot", "warm", "cold-cpu", "cold-disk"]
COLORS = {"hot": "#dc2626", "warm": "#f59e0b", "cold-cpu": "#2563eb", "cold-disk": "#7c3aed"}
pd.set_option("display.width", 220)

def epp(cell):
    rows = []
    for f in glob.glob(f"{cell}/epp_logs/*.log"):
        for line in open(f):
            if '"Request handled"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows.append((pd.Timestamp(d["timestamp"]).value // 1000, d["targetModel"],
                         d["endpoint"].split(":")[0]))
    return pd.DataFrame(rows, columns=["t_us", "adapter", "ip"]).sort_values("t_us")

def load(arm):
    tfs = sorted(glob.glob(f"{base}/{arm}/{WL}/*/trace_data.csv"))
    assert len(tfs) == 1, f"{arm}: expected one iteration, found {len(tfs)}"
    cell = os.path.dirname(tfs[0])
    df = pd.read_csv(tfs[0])
    df = df[df.status == "ok"].copy()
    e = epp(cell)
    # One-to-one match per adapter: the trace's requests are the LAST n EPP decisions
    # for that adapter (earlier ones are warm-up), zipped in send order.
    df["ip"] = None
    warm = []
    for ad, g in df.groupby("adapter"):
        eg = e[e.adapter == ad]
        n = len(g)
        assert len(eg) >= n, f"{arm}/{ad}: {len(eg)} EPP < {n} trace"
        main = eg.iloc[-n:]
        df.loc[g.sort_values("send_time_us").index, "ip"] = main.ip.values
        df.loc[g.sort_values("send_time_us").index, "epp_us"] = main.t_us.values
        warm.append(eg.iloc[:-n])
    warm = pd.concat(warm)
    lag = (df.epp_us - df.send_time_us) / 1000
    print(f"{arm}: send->EPP lag ms p1={lag.quantile(.01):.1f} p50={lag.median():.1f} p99={lag.quantile(.99):.1f}"
          f"  (warm-up EPP decisions used as prior uses: {len(warm)})")
    # EPP clock offset vs client clock, applied to warm-up points
    warm = warm.assign(t_us=warm.t_us - lag.median() * 1000)
    df["ttft_ms"] = (df.first_chunk_time_us - df.send_time_us) / 1000
    pods = {ip: f"pod-{ip.split('.')[-1]}" for ip in df.ip.unique()}
    df["pod"] = df.ip.map(pods)
    warm["pod"] = warm.ip.map(pods)
    return df, warm

def label(df, warm):
    df = df.sort_values("send_time_us").copy()
    lab, qd, rank = [], [], []
    for idx, r in df.iterrows():
        t = r.send_time_us
        same = df[df.pod == r.pod]
        others = same.drop(index=idx)
        executing = others[(others.first_chunk_time_us <= t) & (others.last_chunk_time_us > t)]
        finished = others[others.last_chunk_time_us <= t]
        wu = warm[(warm.pod == r.pod) & (warm.t_us <= t)]
        last_use = pd.concat([
            pd.Series(np.inf, index=executing.adapter.unique()),
            finished.groupby("adapter").last_chunk_time_us.max(),
            wu.groupby("adapter").t_us.max(),
        ]).groupby(level=0).max().sort_values(ascending=False)
        order = list(last_use.index)
        k = order.index(r.adapter) if r.adapter in order else None
        if r.adapter in set(executing.adapter):
            l = "hot"
        elif k is not None and k < GPU_SLOTS:
            l = "warm"
        elif k is not None and k < CPU_SLOTS:
            l = "cold-cpu"
        else:
            l = "cold-disk"
        lab.append(l); rank.append(k)
        qd.append(int(((same.send_time_us < t) & (same.first_chunk_time_us > t)).sum()))
    df["label"], df["queue_at_arrival"], df["recency_rank"] = lab, qd, rank
    return df

data = {}
for arm in arms:
    df, warm = load(arm)
    data[arm] = label(df, warm)

all_ = pd.concat([d.assign(arm=a) for a, d in data.items()])
all_["panel"] = all_.arm + " / " + all_.pod
all_.to_csv(f"{out}/residency_labels.csv", index=False)

# ── Tables ─────────────────────────────────────────────────────────────
print("\nadapters per pod:")
print(all_.groupby("panel").adapter.nunique().to_string())

print("\nTTFT (ms) by label, per arm / pod:")
t = all_.groupby(["panel", "label"]).ttft_ms.agg(n="size", p50="median",
        p90=lambda s: s.quantile(.9)).round(0)
print(t.to_string())

bins = [-1, 0, 10, 50, 1e9]; names = ["0", "1-10", "11-50", ">50"]
all_["queue_bucket"] = pd.cut(all_.queue_at_arrival, bins, labels=names)
print("\nmedian TTFT (ms) by queue depth at arrival x label  [n in brackets]:")
for panel, g in all_.groupby("panel"):
    p = g.pivot_table(index="queue_bucket", columns="label", values="ttft_ms",
                      aggfunc=["median", "size"], observed=False)
    cells = pd.DataFrame({l: [f"{p[('median', l)][b]:.0f} [{int(p[('size', l)][b])}]"
                              if ('median', l) in p and p[('size', l)][b] > 0 else ""
                              for b in names] for l in LABELS if ("size", l) in p}, index=names)
    print(f"-- {panel}"); print(cells.to_string())

print("\nper adapter: label share and median TTFT (ms) by label:")
pa = all_.groupby(["arm", "adapter", "label"]).ttft_ms.agg(["size", "median"]).unstack("label")
share = (pa["size"].div(pa["size"].sum(axis=1), axis=0) * 100).round(0)
med = pa["median"].round(0)
print(pd.concat({"share_%": share, "median_ttft": med}, axis=1).fillna("").to_string())

# ── Charts ─────────────────────────────────────────────────────────────
panels = sorted(all_.panel.unique(), key=lambda p: (p.split(" / ")[0] != "baseline", p))
adapters = list(data["baseline"].adapter.value_counts().index)

# 1. timeline
fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3.2 * len(panels)), sharex=False)
for ax, panel in zip(axes, panels):
    g = all_[all_.panel == panel]; t0 = all_[all_.arm == g.arm.iloc[0]].send_time_us.min()
    for i, ad in enumerate(adapters):
        a = g[g.adapter == ad]
        segs = [((s - t0) / 1e6, (e - s) / 1e6) for s, e in zip(a.first_chunk_time_us, a.last_chunk_time_us)]
        ax.broken_barh(segs, (i - .35, .7), color="#9ca3af", alpha=.25, lw=0)
        for l in LABELS:
            al = a[a.label == l]
            ax.scatter((al.send_time_us - t0) / 1e6, [i] * len(al), s=9, color=COLORS[l],
                       label=l if i == 0 else None, zorder=3)
    ax.set_yticks(range(len(adapters))); ax.set_yticklabels(adapters, fontsize=7)
    ax.set_title(f"{panel}: grey = adapter executing, dots = arrivals by residency label", fontsize=10)
    ax.set_xlabel("seconds since first send"); ax.grid(axis="x", alpha=.3)
handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[l], label=l) for l in LABELS]
fig.legend(handles=handles, loc="upper right", ncol=4)
plt.tight_layout(rect=(0, 0, 1, .98)); plt.savefig(f"{out}/residency_timeline.png", dpi=120); plt.close()

# 2. CDF by label
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), sharey=True)
for ax, panel in zip(axes, panels):
    g = all_[all_.panel == panel]
    for l in LABELS:
        v = np.sort(g[g.label == l].ttft_ms.values) / 1000
        if len(v):
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLORS[l], label=f"{l} (n={len(v)})")
    ax.set_xscale("log"); ax.set_title(panel, fontsize=10); ax.set_xlabel("TTFT (s, log)")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
axes[0].set_ylabel("fraction of requests")
plt.tight_layout(); plt.savefig(f"{out}/residency_ttft_cdf.png", dpi=120); plt.close()

# 3. TTFT vs queue depth
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), sharey=True)
for ax, panel in zip(axes, panels):
    g = all_[all_.panel == panel]
    for l in LABELS:
        gl = g[g.label == l]
        ax.scatter(gl.queue_at_arrival + 1, gl.ttft_ms / 1000, s=8, alpha=.6, color=COLORS[l], label=l)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(panel, fontsize=10)
    ax.set_xlabel("requests queued on pod at arrival + 1 (log)"); ax.grid(alpha=.3)
axes[0].set_ylabel("TTFT (s, log)"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{out}/residency_ttft_vs_queue.png", dpi=120); plt.close()

# 4. per adapter label share
fig, axes = plt.subplots(1, len(arms), figsize=(13, 4.5), sharey=True)
for ax, arm in zip(axes, arms):
    s = share.loc[arm].reindex(adapters)[[l for l in LABELS if l in share.columns]].fillna(0)
    s.plot.barh(stacked=True, ax=ax, color=[COLORS[c] for c in s.columns], width=.75)
    ax.invert_yaxis(); ax.set_xlabel("% of adapter's requests"); ax.set_title(f"{arm}: residency at arrival")
    ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(f"{out}/residency_share_by_adapter.png", dpi=120); plt.close()

for f in ["residency_timeline", "residency_ttft_cdf", "residency_ttft_vs_queue", "residency_share_by_adapter"]:
    print(f"Saved: {out}/{f}.png")
print(f"Saved: {out}/residency_labels.csv")
