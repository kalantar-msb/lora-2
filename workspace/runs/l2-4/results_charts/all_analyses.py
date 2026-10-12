import glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RUN = "l2-4"
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"
BASE = f"{ROOT}/workspace/runs/{RUN}/results"; OUT = f"{ROOT}/workspace/runs/{RUN}/results_charts"
os.makedirs(OUT, exist_ok=True)
arms = sorted(d for d in os.listdir(BASE) if os.path.isdir(f"{BASE}/{d}"))
arms = ["baseline"] + [a for a in arms if a != "baseline"]
COL = {"baseline": "#1f77b4", "routetoholder": "#2ca02c", "loraresidency": "#ff7f0e"}
saved = []
def save(fig, name):
    p = f"{OUT}/{name}"; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); saved.append(p)
pct = lambda s, q: float(np.percentile(s, q)) if len(s) else float("nan")

def traces(arm):
    out = {}
    for wl in sorted(os.listdir(f"{BASE}/{arm}")):
        fs = sorted(glob.glob(f"{BASE}/{arm}/{wl}/*/trace_data.csv")) + sorted(glob.glob(f"{BASE}/{arm}/{wl}/trace_data.csv"))
        if not fs: continue  # plans/ and anything without a trace
        if len(fs) > 1: print(f"note: {arm}/{wl}: pooling {len(fs)} iterations")
        out[wl] = (pd.concat([pd.read_csv(f).assign(iteration=os.path.basename(os.path.dirname(f))) for f in fs], ignore_index=True), fs)
    return out
raw = {a: traces(a) for a in arms}
workloads = sorted(set.intersection(*[set(raw[a]) for a in arms]))
print("arms:", arms, "| workloads:", workloads)

def metrics(df):
    df = df.copy()
    df["ttft_ms"] = (df.first_chunk_time_us - df.send_time_us) / 1e3
    df["e2e_ms"] = (df.last_chunk_time_us - df.send_time_us) / 1e3
    df["tpot_ms"] = np.where(df.output_tokens > 1, (df.last_chunk_time_us - df.first_chunk_time_us) / (df.output_tokens - 1) / 1e3, np.nan)
    df["t_s"] = (df.send_time_us - df.send_time_us.min()) / 1e6
    return df

for wl in workloads:
    full = {a: metrics(raw[a][wl][0]) for a in arms}
    ok = {a: d[d.status == "ok"] for a, d in full.items()}

    # ---- error-rate (catalog: error-rate) ----
    print(f"\n=== error-rate [{wl}] ===")
    rows = []
    for a, d in full.items():
        n = len(d); nok = (d.status == "ok").sum()
        rows.append(dict(arm=a, requests=n, ok=nok, not_ok=n - nok, error_pct=100 * (n - nok) / n,
                         statuses=dict(d.status.value_counts()), finish=dict(d.finish_reason.value_counts())))
    print(pd.DataFrame(rows).to_string(index=False))
    errs = pd.concat([d[d.status != "ok"].assign(arm=a) for a, d in full.items()])
    if len(errs): print(errs.groupby(["arm", "error_message"]).size().to_string())

    # ---- CDFs (catalog: ttft-cdf, tpot-cdf, e2e-cdf) ----
    for m, lab, logx in (("ttft_ms", "TTFT", True), ("tpot_ms", "TPOT", False), ("e2e_ms", "E2E", True)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for a in arms:
            s = np.sort(ok[a][m].dropna().values / (1e3 if logx else 1))
            ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=COL.get(a), label=f"{a} (p50 {pct(s,50):.3g}, p99 {pct(s,99):.3g})")
        if logx: ax.set_xscale("log")
        ax.set_xlabel(f"{lab} ({'s, log' if logx else 'ms'})"); ax.set_ylabel("fraction of requests"); ax.grid(alpha=.3)
        ax.set_title(f"{lab} CDF — {wl}"); ax.legend(fontsize=8)
        save(fig, f"{m[:-3]}_cdf.png")

    # ---- throughput over time (catalog: throughput-over-time) ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    B = 5.0
    for a in arms:
        d = full[a]; t0 = d.send_time_us.min()
        bins = np.arange(0, (d.last_chunk_time_us.max() - t0) / 1e6 + B, B)
        sends, _ = np.histogram((d.send_time_us - t0) / 1e6, bins)
        dn = d[d.status == "ok"]; comp, _ = np.histogram((dn.last_chunk_time_us - t0) / 1e6, bins)
        toks, _ = np.histogram((dn.last_chunk_time_us - t0) / 1e6, bins, weights=dn.output_tokens)
        x = bins[:-1] + B / 2
        axes[0].plot(x, sends / B, color=COL.get(a), label=a); axes[1].plot(x, comp / B, color=COL.get(a), label=a); axes[2].plot(x, toks / B, color=COL.get(a), label=a)
    for ax, t in zip(axes, ("requests sent / s", "requests completed / s", "output tokens / s (completed)")):
        ax.set_ylabel(t); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[-1].set_xlabel(f"seconds since the arm's first send ({B:.0f} s bins)"); axes[0].set_title(f"Throughput over time — {wl}")
    save(fig, "throughput_over_time.png")

    # ---- per-request scatter (catalog: per-request-scatter), TTFT and E2E ----
    fmt = FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else (f"{v:.2f}" if v < 10 else f"{v:.0f}"))
    for m, lab in (("ttft_ms", "TTFT"), ("e2e_ms", "E2E")):
        fig, axes = plt.subplots(len(arms), 1, figsize=(11, 3.2 * len(arms)), sharex=True, sharey=True)
        for ax, a in zip(axes, arms):
            d = ok[a]; ax.scatter(d.t_s, d[m], s=6, alpha=.45, color=COL.get(a))
            ax.set_yscale("log"); ax.yaxis.set_major_formatter(fmt); ax.set_ylabel(f"{lab} (ms, log)"); ax.grid(alpha=.3)
            ax.text(0.01, 0.92, a, transform=ax.transAxes, fontweight="bold")
        axes[-1].set_xlabel("send time (s since the arm's first send)"); axes[0].set_title(f"Per-request {lab} — {wl}")
        save(fig, f"per_request_{m[:-3]}.png")

    # ---- per-adapter latency (extra) ----
    order = ok["baseline"].adapter.value_counts().index.tolist()
    print(f"\n=== per-adapter TTFT p50 / p90 / p99 (ms) and share served < 1 s [{wl}] ===")
    tab = pd.concat({a: ok[a].groupby("adapter").ttft_ms.agg(
        p50=lambda s: pct(s, 50), p90=lambda s: pct(s, 90), p99=lambda s: pct(s, 99),
        fast=lambda s: 100 * (s < 1000).mean()) for a in arms}, axis=1).reindex(order)
    print(tab.to_string(float_format=lambda x: f"{x:,.0f}"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 0.8 / len(arms); xs = np.arange(len(order))
    for i, a in enumerate(arms):
        axes[0].bar(xs + i * w, tab[(a, "p90")] / 1e3, w, color=COL.get(a), label=a)
        axes[1].bar(xs + i * w, tab[(a, "fast")], w, color=COL.get(a), label=a)
    for ax, t in zip(axes, ("TTFT p90 (s)", "% requests with TTFT < 1 s")):
        ax.set_xticks(xs + w * (len(arms) - 1) / 2); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8); ax.set_ylabel(t); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    axes[0].set_title("Per-adapter TTFT p90 (adapters by popularity)"); axes[1].set_title("Per-adapter share served promptly")
    save(fig, "per_adapter_ttft.png")

    # ---- vLLM metrics over time (extra) ----
    def scrape(arm):
        rows = []
        cell = os.path.dirname(raw[arm][wl][1][0])
        for f in sorted(glob.glob(f"{cell}/metrics/raw/*decode*_metrics.log")):
            pod = re.search(r"decode-[^-]+-([^_]+)_", f).group(1); ts = int(re.search(r"_(\d+)_metrics", f).group(1))
            r = dict(pod=pod, t=ts)
            for line in open(f, errors="replace"):
                if line.startswith("#"): continue
                for key, pat in (("running", "vllm:num_requests_running{"), ("waiting", "vllm:num_requests_waiting{"), ("kv", "vllm:kv_cache_usage_perc{"),
                                 ("gen_tokens", "vllm:generation_tokens_total{"), ("preempt", "vllm:num_preemptions_total{")):
                    if line.startswith(pat): r[key] = float(line.split()[-1])
                if line.startswith("vllm:num_requests_waiting_by_reason") and 'reason="deferred"' in line: r["deferred"] = float(line.split()[-1])
            rows.append(r)
        df = pd.DataFrame(rows).sort_values(["pod", "t"])
        df["gen_tps"] = df.groupby("pod").gen_tokens.diff() / df.groupby("pod").t.diff()
        t0 = (raw[arm][wl][0].send_time_us.min()) / 1e6
        df["t_rel"] = df.t - t0
        return df
    ms = {a: scrape(a) for a in arms}
    fig, axes = plt.subplots(4, len(arms), figsize=(5 * len(arms), 11), sharex=True, sharey="row")
    for j, a in enumerate(arms):
        for pod, g in ms[a].groupby("pod"):
            for i, k in enumerate(("running", "deferred", "kv", "gen_tps")):
                axes[i, j].plot(g.t_rel, g[k] * (100 if k == "kv" else 1), marker=".", label=pod)
        axes[0, j].set_title(a)
        print(f"[{a}] metrics: " + "; ".join(f"{p}: running max {g.running.max():.0f}, deferred max {g.deferred.max():.0f}, KV max {100*g.kv.max():.1f}%, preemptions {g.preempt.max()-g.preempt.min():.0f}" for p, g in ms[a].groupby("pod")))
    for i, t in enumerate(("requests running", "requests deferred (no adapter slot)", "KV cache use (%)", "generation tokens / s")):
        axes[i, 0].set_ylabel(t)
    for ax in axes.flat: ax.grid(alpha=.3); ax.legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel("s since the arm's first send (15 s scrapes)")
    save(fig, "vllm_metrics_over_time.png")

for p in saved: print("Saved:", p)
