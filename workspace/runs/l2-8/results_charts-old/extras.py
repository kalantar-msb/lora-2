# Per-server distribution, per-adapter TTFT p50/p90/p99, per-server TPOT vs concurrency and the
# E2E split, for each repeat. Reuses cells()/load_cell() from analysis.py.
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"; OUT = f"{ROOT}/workspace/runs/l2-8/results_charts"
ns = {}; exec(open(f"{OUT}/analysis.py").read().split("C = {a: cells(a) for a in arms}")[0], ns)
arms, cells, load_cell, pct, COL, TAG = ns["arms"], ns["cells"], ns["load_cell"], ns["pct"], ns["COL"], ns["TAG"]
order = [c["adapter"] for c in yaml.safe_load(open(f"{ROOT}/workloads/granite-27adapter.workload.yaml"))["clients"]]
for wl, tag in TAG.items():
    D = {a: load_cell(cells(a)[wl][0]) for a in arms}
    ok = {a: D[a]["tr"][D[a]["tr"].status == "ok"].copy() for a in arms}
    print(f"\n######## {tag} ({wl}) ########")
    # --- E2E split: waiting (TTFT) vs decoding (E2E - TTFT) ---
    print("E2E split (ms, mean):")
    for a in arms:
        t = ok[a]; print(f"  {a:<14} TTFT {t.ttft_ms.mean():7.1f}  decode {(t.e2e_ms - t.ttft_ms).mean():7.1f}  E2E {t.e2e_ms.mean():7.1f}  mean output tokens {t.output_tokens.mean():.1f}")
    # --- per-server table: share, TPOT, concurrency from metrics ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5)); w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        t, met = ok[a], D[a]["met"]; g = t.groupby("pod")
        df = pd.DataFrame({"requests": g.size(), "share_pct": 100 * g.size() / len(t), "adapters": g.adapter.nunique(),
                           "ttft_p50": g.ttft_ms.median(), "ttft_p99": g.ttft_ms.quantile(.99),
                           "tpot_p50": g.tpot_ms.median(), "tpot_p90": g.tpot_ms.quantile(.9),
                           "running_mean": met.groupby("pod").running.mean(), "running_max": met.groupby("pod").running.max()}).sort_values("requests", ascending=False)
        cv = df.requests.std(ddof=0) / df.requests.mean()
        print(f"\n[{a}] per server (CV of requests {cv:.2f}; even share 25%)\n" + df.round(1).to_string())
        xs = np.arange(len(df))
        for ax, col, lab in zip(axes[:2], ("share_pct", "ttft_p99"), ("% of requests", "TTFT p99 (ms, log)")):
            bars = ax.bar(xs + i * w, df[col].values, w, color=COL.get(a), label=a)
            for b, p in zip(bars, df.index): ax.text(b.get_x() + b.get_width() / 2, b.get_height(), p, ha="center", va="bottom", fontsize=6, rotation=90)
            ax.set_ylabel(lab); ax.set_xticks([]); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
        axes[2].scatter(df.running_mean, df.tpot_p50, s=60, color=COL.get(a), label=a)
        for p, r in df.iterrows(): axes[2].annotate(p, (r.running_mean, r.tpot_p50), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].axhline(25, color="k", ls=":", lw=1); axes[1].set_yscale("log")
    axes[2].set_xlabel("mean requests running (15 s scrapes)"); axes[2].set_ylabel("TPOT p50 (ms)"); axes[2].grid(alpha=.3); axes[2].legend(fontsize=8)
    axes[0].set_title(f"{tag}: request share per server (busiest first)"); axes[1].set_title("TTFT p99 per server"); axes[2].set_title("TPOT vs load, per server")
    fig.tight_layout(); p = f"{OUT}/{tag}_request_distribution_between_servers.png"; fig.savefig(p, dpi=120); plt.close(fig); print("Saved:", p)
    # --- per-adapter TTFT p50/p90/p99 ---
    tab = {a: ok[a].groupby("adapter").ttft_ms.agg(n="count", p50=lambda s: pct(s, 50), p90=lambda s: pct(s, 90), p99=lambda s: pct(s, 99)).reindex(order) for a in arms}
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True); xs = np.arange(len(order))
    for ax, q in zip(axes, ("p50", "p90", "p99")):
        for i, a in enumerate(arms):
            ax.bar(xs + i * w, tab[a][q] / 1e3, w, color=COL.get(a), label=f"{a} (all requests {q}: {pct(ok[a].ttft_ms, int(q[1:]))/1e3:.3g} s)")
        ax.set_yscale("log"); ax.set_ylabel(f"TTFT {q} (s, log)"); ax.grid(axis="y", alpha=.3, which="both"); ax.legend(fontsize=8, loc="upper left")
    lower = {q: int((tab["loraresidency"][q] < tab["baseline"][q]).sum()) for q in ("p50", "p90", "p99")}
    print("adapters where loraresidency is lower: " + ", ".join(f"{q} {v}/27" for q, v in lower.items()))
    axes[-1].set_xticks(xs + w * (len(arms) - 1) / 2); axes[-1].set_xticklabels([f"{x} ({int(tab[arms[0]].loc[x, 'n'])})" for x in order], rotation=60, ha="right", fontsize=7)
    axes[0].set_title(f"{tag}: per-adapter TTFT p50 / p90 / p99 — adapters by popularity (request count in brackets)")
    fig.tight_layout(); p = f"{OUT}/{tag}_per_adapter_ttft.png"; fig.savefig(p, dpi=120); plt.close(fig); print("Saved:", p)
