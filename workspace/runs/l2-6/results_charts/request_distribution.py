"""Request distribution per arm and pod: which server each request went to.

Sources, per arm:
  - EPP "Request handled" lines: one per routing decision, naming targetModel and the
    chosen endpoint (pod IP). Split into the trace window (requests whose time falls
    within the trace's first..last send) and everything else (prewarm, warmup, probes).
  - vLLM server logs: one "POST /v1/completions" line per request served, per pod,
    over the whole log. Independent of the EPP, used as a cross-check.
"""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "l2-6"
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"
BASE = f"{ROOT}/workspace/runs/{RUN}/results"; OUT = f"{ROOT}/workspace/runs/{RUN}/results_charts"
WL = "granite-9adapter.workload"
arms = sorted(d for d in os.listdir(BASE) if os.path.isdir(f"{BASE}/{d}"))
arms = ["baseline"] + [a for a in arms if a != "baseline"]


def cells(arm):
    ds = sorted(glob.glob(f"{BASE}/{arm}/{WL}/i*/")) or [f"{BASE}/{arm}/{WL}/"]
    if len(ds) > 1:
        print(f"note: {arm} has {len(ds)} iterations; pooling all of them")
    return ds


def load(arm):
    eps, served, traces = [], [], []
    for d in cells(arm):
        it = os.path.basename(d.rstrip("/"))
        tr = pd.read_csv(f"{d}/trace_data.csv"); tr = tr[tr.status == "ok"]
        traces.append(tr.assign(iteration=it))
        lo, hi = tr.send_time_us.min() - 50_000, tr.send_time_us.max() + 50_000
        ipmap = {}
        for f in glob.glob(f"{d}/resources/pod-*decode*.yaml"):
            t = open(f).read()
            ip = re.search(r"podIP: (\S+)", t); nm = re.search(r"^  name: (\S+)", t, re.M)
            if ip and nm:
                ipmap[ip.group(1)] = nm.group(1).rsplit("-", 1)[-1]
        for f in sorted(glob.glob(f"{d}/epp_logs/*.log")):
            for line in open(f, errors="replace"):
                if '"Request handled"' not in line:
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                t_us = pd.Timestamp(j["timestamp"]).value // 1000
                model = j.get("targetModel") or j.get("targetModelName")
                pod = ipmap.get(str(j.get("endpoint", "")).split(":")[0], j.get("endpoint"))
                eps.append(dict(iteration=it, t_us=t_us, adapter=model, pod=pod,
                                window=bool(lo <= t_us <= hi)))
        for f in glob.glob(f"{d}/server_logs/*.log"):
            pod = re.search(r"decode-[^-]+-([^_]+)_", f).group(1)
            served.append(dict(iteration=it, pod=pod, served=sum('POST /v1/completions' in l for l in open(f, errors="replace"))))
    ep = pd.DataFrame(eps)
    base_model = traces[0].model.iloc[0] if "model" in traces[0] else None
    ep["kind"] = np.where(ep.adapter == base_model, "base model", "adapter")
    return ep, pd.DataFrame(served), pd.concat(traces, ignore_index=True)


data = {a: load(a) for a in arms}
all_ads = data["baseline"][2].adapter.value_counts().index.tolist()

print(f"=== Request distribution per pod — run {RUN} ===")
summary = []
for a, (ep, sv, tr) in data.items():
    w = ep[ep.window & (ep.kind == "adapter")]
    per_pod = w.pod.value_counts().sort_index()
    n = per_pod.values.astype(float)
    jain = n.sum() ** 2 / (len(n) * (n ** 2).sum())
    other = ep[~(ep.window & (ep.kind == "adapter"))].groupby(["pod", "kind"]).size().unstack(fill_value=0)
    allep = ep.pod.value_counts()
    srv = sv.groupby("pod").served.sum()
    tab = pd.DataFrame({"trace-window adapter reqs": per_pod, "share %": 100 * per_pod / per_pod.sum()})
    tab["adapters served"] = w.groupby("pod").adapter.nunique()
    for k in ("adapter", "base model"):
        tab[f"outside window: {k}"] = other.get(k, pd.Series(dtype=int))
    tab["EPP total"] = allep; tab["vLLM POSTs (server log)"] = srv
    tab = tab.fillna(0)
    print(f"\n[{a}]  trace rows {len(tr)}; routed adapter requests in trace window {int(n.sum())}")
    print(tab.to_string(float_format=lambda x: f"{x:,.1f}"))
    print(f"  balance: max/mean {n.max() / n.mean():.2f}, CV {n.std() / n.mean():.2f}, Jain {jain:.3f} (1.0 = even)")
    mism = (tab["EPP total"] - tab["vLLM POSTs (server log)"]).abs()
    print(f"  EPP-vs-vLLM per-pod count difference: max {int(mism.max())}")
    summary.append(dict(arm=a, pods=len(n), min=int(n.min()), max=int(n.max()), max_over_mean=n.max() / n.mean(), cv=n.std() / n.mean(), jain=jain))
print("\n=== Balance summary (trace-window adapter requests) ===")
print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Chart 1: stacked bars, requests per pod by adapter (trace window)
cmap = plt.get_cmap("tab10"); col = {ad: cmap(i % 10) for i, ad in enumerate(all_ads)}
fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 5), sharey=True)
for ax, a in zip(axes, arms):
    w = data[a][0]; w = w[w.window & (w.kind == "adapter")]
    m = w.pivot_table(index="pod", columns="adapter", values="t_us", aggfunc="count", fill_value=0).reindex(columns=all_ads, fill_value=0)
    bottom = np.zeros(len(m))
    for ad in all_ads:
        ax.bar(m.index, m[ad], bottom=bottom, color=col[ad], label=ad); bottom += m[ad].values
    for x, tot in zip(m.index, bottom):
        ax.text(x, tot, f"{int(tot)}", ha="center", va="bottom", fontsize=8)
    ax.set_title(a); ax.set_xlabel("pod"); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("requests (trace window)"); axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.suptitle(f"Requests per pod, by adapter — {RUN}")
p1 = f"{OUT}/request_distribution_by_pod.png"; fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)

# Chart 2: per-pod request rate over time (trace window, 10 s bins)
B = 10.0
fig, axes = plt.subplots(len(arms), 1, figsize=(11, 3 * len(arms)), sharex=True, sharey=True)
for ax, a in zip(axes, arms):
    w = data[a][0]; w = w[w.window & (w.kind == "adapter")]
    t0 = data[a][2].send_time_us.min()
    bins = np.arange(0, (w.t_us.max() - t0) / 1e6 + B, B)
    for pod, g in w.groupby("pod"):
        h, _ = np.histogram((g.t_us - t0) / 1e6, bins); ax.plot(bins[:-1] + B / 2, h / B, marker=".", label=pod)
    ax.set_ylabel("req/s"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper right"); ax.text(0.01, 0.88, a, transform=ax.transAxes, fontweight="bold")
axes[-1].set_xlabel(f"seconds since the arm's first send ({B:.0f} s bins)"); axes[0].set_title(f"Requests routed to each pod over time — {RUN}")
p2 = f"{OUT}/request_rate_per_pod.png"; fig.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
for p in (p1, p2):
    print("Saved:", p)
