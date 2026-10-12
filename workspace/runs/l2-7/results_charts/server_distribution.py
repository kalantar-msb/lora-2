import glob, re, os, numpy as np, pandas as pd, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"; RUN = "l2-7"
BASE = f"{ROOT}/workspace/runs/{RUN}/results"; OUT = f"{ROOT}/workspace/runs/{RUN}/results_charts"
spec = importlib.util.spec_from_file_location("a", f"{OUT}/analysis.py"); src = open(f"{OUT}/analysis.py").read()
# reuse load_cell / cells from the saved analysis script without re-running its report
ns = {}; exec(src.split("C = {a: cells(a) for a in arms}")[0], ns)
arms, cells, load_cell, pct = ns["arms"], ns["cells"], ns["load_cell"], ns["pct"]
COL = ns["COL"]
rows, per_arm = [], {}
for a in arms:
    for wl, ds in cells(a).items():
        for d in ds:
            c = load_cell(d); t = c["tr"][c["tr"].status == "ok"]
            # independent count: completion requests logged by each vLLM pod (includes warm-up/prewarm/preflight traffic)
            srv = {}
            for f in glob.glob(f"{d}/server_logs/*.log"):
                pod = re.search(r"decode-[^-]+-([^_]+)_", f).group(1)
                srv[pod] = sum(1 for line in open(f, errors="replace") if 'POST /v1/completions HTTP/1.1" 200' in line)
            g = t.groupby("pod")
            df = pd.DataFrame({"requests": g.size(), "share_pct": 100 * g.size() / len(t), "adapters": g.adapter.nunique(),
                               "output_tok_s": g.output_tokens.sum() / ((t.last_chunk_time_us.max() - t.send_time_us.min()) / 1e6),
                               "ttft_p50": g.ttft_ms.median(), "ttft_p99": g.ttft_ms.quantile(.99), "e2e_p99": g.e2e_ms.quantile(.99),
                               "server_log_200s": pd.Series(srv)})
            n = len(df); mean = df.requests.mean()
            spread = 100 * (df.requests.max() - df.requests.min()) / mean
            cv = df.requests.std(ddof=0) / mean
            print(f"\n=== Distribution between servers — {a} ({wl}, {os.path.basename(d.rstrip('/'))}) ===")
            print(df.round(1).to_string())
            print(f"  spread (max-min as % of mean) {spread:.0f}%; CV {cv:.2f}; even share would be {100/n:.0f}% each; "
                  f"server-log 200s minus trace rows = {int(df.server_log_200s.sum() - len(t))} (warm-up, prewarm, preflight)")
            per_arm[a] = df
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
xs = np.arange(max(len(d) for d in per_arm.values())); w = 0.8 / len(arms)
for i, a in enumerate(arms):
    d = per_arm[a].sort_values("requests", ascending=False)
    for ax, col, lab in zip(axes, ("share_pct", "adapters", "ttft_p99"), ("% of requests", "distinct adapters served", "TTFT p99 (ms)")):
        bars = ax.bar(xs[:len(d)] + i * w, d[col].values, w, color=COL.get(a), label=a)
        for b, p in zip(bars, d.index): ax.text(b.get_x() + b.get_width() / 2, b.get_height(), p, ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_ylabel(lab); ax.set_xticks([]); ax.grid(axis="y", alpha=.3)
axes[0].axhline(100 / len(xs), color="k", ls=":", lw=1, label="even split"); axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
axes[2].set_yscale("log")
axes[0].set_title("Request share per server (busiest first)"); axes[1].set_title("Adapters each server served"); axes[2].set_title("TTFT p99 per server")
fig.tight_layout(); p = f"{OUT}/request_distribution_between_servers.png"; fig.savefig(p, dpi=120); print("\nSaved:", p)
