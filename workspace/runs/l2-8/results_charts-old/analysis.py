import glob, json, math, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RUN = "l2-8"
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"
BASE = f"{ROOT}/workspace/runs/{RUN}/results"; OUT = f"{ROOT}/workspace/runs/{RUN}/results_charts"
os.makedirs(OUT, exist_ok=True)
arms = sorted(d for d in os.listdir(BASE) if os.path.isdir(f"{BASE}/{d}")); arms = ["baseline"] + [a for a in arms if a != "baseline"]
COL = {"baseline": "#1f77b4", "loraresidency": "#ff7f0e", "routetoholder": "#2ca02c"}
pct = lambda s, q: float(np.percentile(s, q)) if len(s) else float("nan")
saved = []
# both cells ran 15 req/s (the -30 workload file was edited after assembly), so they are two repeats
TAG = {"granite-27adapter.workload": "repA", "granite-27adapter.workload-30": "repB"}
def save(fig, name):
    p = f"{OUT}/{TAG.get(wl, wl)}_{name}"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); saved.append(p)

def cells(arm):
    out = {}
    for wl in sorted(os.listdir(f"{BASE}/{arm}")):
        ds = sorted(glob.glob(f"{BASE}/{arm}/{wl}/i*/")) or ([f"{BASE}/{arm}/{wl}/"] if os.path.exists(f"{BASE}/{arm}/{wl}/trace_data.csv") else [])
        if ds: out[wl] = ds
    return out

def load_cell(d):
    tr = pd.read_csv(f"{d}/trace_data.csv")
    tr["ttft_ms"] = (tr.first_chunk_time_us - tr.send_time_us) / 1e3
    tr["e2e_ms"] = (tr.last_chunk_time_us - tr.send_time_us) / 1e3
    tr["tpot_ms"] = np.where(tr.output_tokens > 1, (tr.last_chunk_time_us - tr.first_chunk_time_us) / (tr.output_tokens - 1) / 1e3, np.nan)
    tr["send_time_us"] = tr.send_time_us.astype("int64")
    ipmap = {}
    for f in glob.glob(f"{d}/resources/pod-*decode*.yaml"):
        t = open(f).read(); ip = re.search(r"podIP: (\S+)", t); nm = re.search(r"^  name: (\S+)", t, re.M)
        if ip and nm: ipmap[ip.group(1)] = nm.group(1).rsplit("-", 1)[-1]
    recs, rem = [], {}
    for f in sorted(glob.glob(f"{d}/epp_logs/*.log")):
        for line in open(f, errors="replace"):
            if '"Completed running filter plugins"' in line:
                try: j = json.loads(line); rem[j.get("x-request-id")] = j.get("remainingEndpoints")
                except Exception: pass
            elif '"Request handled"' in line:
                try: j = json.loads(line)
                except Exception: continue
                recs.append(dict(t_us=int(pd.Timestamp(j["timestamp"]).value // 1000), rid=j.get("x-request-id"), adapter=j.get("targetModel"),
                                 pod=ipmap.get(j.get("endpoint", "").split(":")[0], j.get("endpoint"))))
    ep = pd.DataFrame(recs); ep["remaining"] = ep.rid.map(rem)
    j = pd.merge_asof(tr.sort_values("send_time_us"), ep.sort_values("t_us").rename(columns={"t_us": "epp_t_us"}), left_on="send_time_us", right_on="epp_t_us",
                      left_by="adapter", right_by="adapter", direction="nearest", tolerance=200_000)
    lag = (j.epp_t_us - j.send_time_us).abs() / 1e3
    note = f"join {j.pod.notna().sum()}/{len(j)} rows to EPP decisions, |EPP-send| p50 {lag.median():.1f} ms p99 {pct(lag.dropna(),99):.1f} ms"
    mrows = []
    for f in sorted(glob.glob(f"{d}/metrics/raw/*decode*_metrics.log")):
        pod = re.search(r"decode-[^-]+-([^_]+)_", f).group(1); ts = int(re.search(r"_(\d+)_metrics", f).group(1)) * 1e6
        r = dict(pod=pod, t_us=ts)
        for line in open(f, errors="replace"):
            if line.startswith("#"): continue
            for key, pat in (("running", "vllm:num_requests_running{"), ("waiting", "vllm:num_requests_waiting{"), ("kv", "vllm:kv_cache_usage_perc{"),
                             ("gen_tokens", "vllm:generation_tokens_total{"), ("preempt", "vllm:num_preemptions_total{")):
                if line.startswith(pat): r[key] = float(line.split()[-1])
            if line.startswith("vllm:num_requests_waiting_by_reason") and 'reason="deferred"' in line: r["deferred"] = float(line.split()[-1])
        mrows.append(r)
    met = pd.DataFrame(mrows).sort_values(["pod", "t_us"])
    met["gen_tps"] = met.groupby("pod").gen_tokens.diff() / (met.groupby("pod").t_us.diff() / 1e6)
    srows = []
    for f in glob.glob(f"{d}/server_logs/*.log"):
        pod = re.search(r"decode-[^-]+-([^_]+)_", f).group(1)
        for line in open(f, errors="replace"):
            m = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers.py.*Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?", line)
            if m: srows.append(dict(pod=pod, t_us=pd.Timestamp(f"2026-{m.group(1)}", tz="UTC").value / 1e3, deferred=int(m.group(4) or 0)))
    txt = " ".join(open(f, errors="replace").read() for f in glob.glob(f"{d}/server_logs/*.log"))
    slots = max(int(x) for x in re.findall(r"'max_loras': (\d+)", txt)); cpu = max(int(x) for x in re.findall(r"'max_cpu_loras': (\d+)", txt))
    return dict(tr=j, met=met, st=pd.DataFrame(srows), slots=slots, cpu=cpu, note=note)

C = {a: cells(a) for a in arms}
workloads = sorted(set.intersection(*[set(C[a]) for a in arms]))
print("arms:", arms, "| workloads:", workloads)
for wl in workloads:
    D = {}
    for a in arms:
        parts = [load_cell(d) for d in C[a][wl]]
        if len(parts) > 1: print(f"note: {a}/{wl}: pooling {len(parts)} iterations for latency; residency uses each separately")
        D[a] = parts[0] | {"tr": pd.concat([p["tr"] for p in parts], ignore_index=True)}
        print(f"[{a}] slots/pod {D[a]['slots']}, CPU cache/pod {D[a]['cpu']}, {D[a]['note']}")
    ok = {a: D[a]["tr"][D[a]["tr"].status == "ok"].copy() for a in arms}
    for a in arms: ok[a]["t_s"] = (ok[a].send_time_us - ok[a].send_time_us.min()) / 1e6

    # ---------- error rate ----------
    print("\n=== Error rate ===")
    for a in arms:
        t = D[a]["tr"]; print(f"  {a:<14} {len(t)} requests, ok {(t.status=='ok').sum()}, not ok {(t.status!='ok').sum()}  finish={dict(t.finish_reason.value_counts())}")
    # ---------- latency table ----------
    print("\n=== Latency (status=ok; ms) ===")
    rows = []
    for a in arms:
        for m in ("ttft_ms", "tpot_ms", "e2e_ms"):
            s = ok[a][m].dropna(); rows.append(dict(arm=a, metric=m[:-3].upper(), mean=s.mean(), p50=pct(s,50), p90=pct(s,90), p99=pct(s,99)))
        rows.append(dict(arm=a, metric="TTFT<1s %", mean=100*(ok[a].ttft_ms<1000).mean(), p50=np.nan, p90=np.nan, p99=np.nan))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # ---------- request distribution ----------
    spec = __import__("yaml").safe_load(open(f"{ROOT}/workloads/{wl}.yaml")) if os.path.exists(f"{ROOT}/workloads/{wl}.yaml") else None
    expected = {c["adapter"]: c["rate_fraction"] for c in spec["clients"]} if spec else {}
    order = list(expected) if expected else ok["baseline"].adapter.value_counts().index.tolist()
    print("\n=== Request distribution: per-adapter share vs rate_fraction (client side) ===")
    for a in arms:
        n = len(ok[a]); obs = ok[a].adapter.value_counts().reindex(order).fillna(0)
        exp = np.array([expected.get(x, np.nan) for x in order]) * n
        chi = float(np.nansum((obs.values - exp) ** 2 / exp)); dfree = len(order) - 1
        z = (obs.values - exp) / np.sqrt(exp * (1 - exp / n))
        # chi-square p-value via Wilson-Hilferty approximation
        pz = ((chi / dfree) ** (1/3) - (1 - 2/(9*dfree))) / math.sqrt(2/(9*dfree)); p = 0.5 * math.erfc(pz / math.sqrt(2))
        print(f"  [{a}] adapters seen {int((obs>0).sum())}/{len(order)}; chi2={chi:.1f} df={dfree} p~{p:.3f}; |z|>3: {int((np.abs(z)>3).sum())}; max |z| {np.nanmax(np.abs(z)):.2f}")
    print("\n=== Request distribution: adapter x pod (requests) and pods used per adapter ===")
    conc = {}
    for a in arms:
        mat = ok[a].pivot_table(index="adapter", columns="pod", values="request_id", aggfunc="count", fill_value=0).reindex(order).fillna(0).astype(int)
        top = (mat.max(axis=1) / mat.sum(axis=1).replace(0, np.nan))
        npods = (mat > 0).sum(axis=1)
        conc[a] = (mat, top, npods)
        w = (mat.sum(axis=1) / mat.values.sum())
        print(f"\n[{a}] per-pod totals: " + ", ".join(f"{p}: {int(v)} ({100*v/mat.values.sum():.1f}%)" for p, v in mat.sum().items())
              + f"; traffic-weighted top-pod share {100*(top*w).sum():.1f}%; adapters on 1 pod: {int((npods==1).sum())}, on 2: {int((npods==2).sum())}, on 3: {int((npods==3).sum())}, on 4: {int((npods==4).sum())}")
        print(mat.assign(top_pod_pct=(100*top).round(0)).to_string())
        rr = ok[a].remaining.value_counts().sort_index(); print("  candidate pods left after filters: " + ", ".join(f"{int(k)}: {v}" for k, v in rr.items()))

    fig, axes = plt.subplots(1, len(arms), figsize=(5.5 * len(arms), 10))
    for ax, a in zip(np.atleast_1d(axes), arms):
        mat = conc[a][0]; share = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
        im = ax.imshow(share.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(mat))); ax.set_yticklabels([f"{x} ({int(n)})" for x, n in zip(mat.index, mat.sum(axis=1))], fontsize=7)
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, fontsize=8)
        for i in range(len(mat)):
            for k in range(len(mat.columns)):
                v = mat.values[i, k]
                if v: ax.text(k, i, str(v), ha="center", va="center", fontsize=6, color="w" if share.values[i, k] < .6 else "k")
        ax.set_title(f"{a}: requests per adapter x pod\n(colour = share of the adapter's requests; adapters by popularity)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=.5, label="share of adapter's requests on the pod"); save(fig, "request_distribution_adapter_by_pod.png")

    fig, axes = plt.subplots(len(arms), 1, figsize=(12, 3.3 * len(arms)), sharex=True)
    for ax, a in zip(np.atleast_1d(axes), arms):
        t = ok[a]; bins = np.arange(0, t.t_s.max() + 10, 10)
        for p, g in t.groupby("pod"):
            h, _ = np.histogram(g.t_s, bins); ax.plot(bins[:-1] + 5, h / 10, label=f"{p} ({len(g)})")
        ax.set_ylabel("requests / s"); ax.set_title(f"{a}: requests routed to each pod over time (10 s bins)", fontsize=9); ax.legend(fontsize=7, ncol=4); ax.grid(alpha=.3)
    np.atleast_1d(axes)[-1].set_xlabel("seconds since the arm's first send"); save(fig, "request_distribution_per_pod_over_time.png")

    fig, ax = plt.subplots(figsize=(13, 4.5)); xs = np.arange(len(order)); wbar = 0.8 / (len(arms) + 1)
    if expected: ax.bar(xs, [100 * expected[x] for x in order], wbar, color="#999", label="rate_fraction (spec)")
    for i, a in enumerate(arms):
        obs = ok[a].adapter.value_counts().reindex(order).fillna(0); ax.bar(xs + (i + 1) * wbar, 100 * obs / obs.sum(), wbar, color=COL.get(a), label=a)
    ax.set_yscale("log"); ax.set_xticks(xs + wbar); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7); ax.set_ylabel("% of requests (log)")
    ax.set_title("Per-adapter request share: spec vs observed"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); save(fig, "request_distribution_adapter_share.png")

    # ---------- CDFs ----------
    for m, lab, logx in (("ttft_ms", "TTFT", True), ("tpot_ms", "TPOT", False), ("e2e_ms", "E2E", True)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for a in arms:
            s = np.sort(ok[a][m].dropna().values / (1e3 if logx else 1)); ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=COL.get(a), label=f"{a} (p50 {pct(s,50):.3g}, p99 {pct(s,99):.3g})")
        if logx: ax.set_xscale("log")
        ax.set_xlabel(f"{lab} ({'s, log' if logx else 'ms'})"); ax.set_ylabel("fraction of requests"); ax.grid(alpha=.3); ax.set_title(f"{lab} CDF — {wl}"); ax.legend(fontsize=8)
        save(fig, f"{m[:-3]}_cdf.png")
    # ---------- throughput ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True); B = 10.0
    for a in arms:
        d = D[a]["tr"]; t0 = d.send_time_us.min(); bins = np.arange(0, (d.last_chunk_time_us.max() - t0) / 1e6 + B, B); dn = d[d.status == "ok"]
        for ax, (h, _) in zip(axes, (np.histogram((d.send_time_us - t0) / 1e6, bins), np.histogram((dn.last_chunk_time_us - t0) / 1e6, bins), np.histogram((dn.last_chunk_time_us - t0) / 1e6, bins, weights=dn.output_tokens))):
            ax.plot(bins[:-1] + B / 2, h / B, color=COL.get(a), label=a)
    for ax, t in zip(axes, ("requests sent / s", "requests completed / s", "output tokens / s")): ax.set_ylabel(t); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[-1].set_xlabel(f"seconds since the arm's first send ({B:.0f} s bins)"); axes[0].set_title(f"Throughput over time — {wl}"); save(fig, "throughput_over_time.png")
    # ---------- per-request scatter, one figure per arm, coloured by pod ----------
    fmt = FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else (f"{v:.2f}" if v < 10 else f"{v:.0f}"))
    for m, lab in (("ttft_ms", "TTFT"), ("e2e_ms", "E2E")):
        fig, axes = plt.subplots(len(arms), 1, figsize=(12, 3.4 * len(arms)), sharex=True, sharey=True)
        for ax, a in zip(np.atleast_1d(axes), arms):
            for p, g in ok[a].groupby("pod"): ax.scatter(g.t_s, g[m], s=4, alpha=.4, label=p)
            ax.set_yscale("log"); ax.yaxis.set_major_formatter(fmt); ax.set_ylabel(f"{lab} (ms, log)"); ax.grid(alpha=.3); ax.legend(fontsize=7, markerscale=3, ncol=4, loc="upper right")
            ax.text(0.01, 0.92, a, transform=ax.transAxes, fontweight="bold")
        np.atleast_1d(axes)[-1].set_xlabel("send time (s since the arm's first send)"); np.atleast_1d(axes)[0].set_title(f"Per-request {lab}, coloured by pod — {wl}"); save(fig, f"per_request_{m[:-3]}.png")
    # ---------- per-adapter latency ----------
    print("\n=== Per-adapter TTFT p50 / p90 / p99 (ms) and % < 1 s ===")
    tab = pd.concat({a: ok[a].groupby("adapter").ttft_ms.agg(n="count", p50=lambda s: pct(s,50), p90=lambda s: pct(s,90), p99=lambda s: pct(s,99), fast=lambda s: 100*(s<1000).mean()) for a in arms}, axis=1).reindex(order)
    print(tab.to_string(float_format=lambda x: f"{x:,.0f}"))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True); xs = np.arange(len(order)); wb = 0.8 / len(arms)
    for i, a in enumerate(arms):
        axes[0].bar(xs + i * wb, tab[(a, "p90")] / 1e3, wb, color=COL.get(a), label=a); axes[1].bar(xs + i * wb, tab[(a, "fast")], wb, color=COL.get(a), label=a)
    axes[0].set_ylabel("TTFT p90 (s)"); axes[1].set_ylabel("% TTFT < 1 s"); axes[1].set_xticks(xs + wb * (len(arms) - 1) / 2); axes[1].set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    for ax in axes: ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    axes[0].set_title("Per-adapter TTFT (adapters by popularity)"); save(fig, "per_adapter_ttft.png")
    # ---------- vLLM metrics, one figure per arm ----------
    for a in arms:
        met = D[a]["met"]; t0 = ok[a].send_time_us.min()
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for p, g in met.groupby("pod"):
            for ax, k in zip(axes, ("running", "deferred", "kv", "gen_tps")): ax.plot((g.t_us - t0) / 1e6, g[k] * (100 if k == "kv" else 1), marker=".", label=p)
        for ax, t in zip(axes, ("requests running", "requests deferred (no adapter slot)", "KV cache use (%)", "generation tokens / s")): ax.set_ylabel(t); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=4)
        axes[0].set_title(f"{a}: vLLM metrics per pod (15 s scrapes)"); axes[-1].set_xlabel("s since the arm's first send"); save(fig, f"vllm_metrics_{a}.png")
        print(f"[{a}] metrics: " + "; ".join(f"{p}: running max {g.running.max():.0f}, deferred max {g.deferred.max():.0f}, KV max {100*g.kv.max():.1f}%, preempt {g.preempt.max()-g.preempt.min():.0f}" for p, g in met.groupby("pod")))
    # ---------- residency: reconstruction + two-tier replay, one figure per arm ----------
    print("\n=== Residency (reconstructed from first-token..last-token intervals; two-tier LRU replay) ===")
    for a in arms:
        t = ok[a]; slots, cpu = D[a]["slots"], max(D[a]["cpu"], D[a]["slots"]); pods = sorted(t.pod.dropna().unique())
        fig, axes = plt.subplots(len(pods), 1, figsize=(13, 4.2 * len(pods)), sharex=True); t0 = t.send_time_us.min()
        for ax, p in zip(np.atleast_1d(axes), pods):
            g = t[t.pod == p]; ev = sorted([(r.first_chunk_time_us, 1, r.adapter) for r in g.itertuples()] + [(r.last_chunk_time_us, -1, r.adapter) for r in g.itertuples()], key=lambda x: (x[0], x[1]))
            cnt, order_r, copies, disk, loads, maxset, on, spans = {}, [], 0, 0, 0, 0, {}, []
            for ts, k, ad in ev:
                if k == 1:
                    if ad in order_r:
                        i = order_r.index(ad)
                        if i >= slots: loads += 1; copies += i < cpu; disk += i >= cpu
                        order_r.pop(i)
                    else: loads += 1; disk += 1
                    order_r.insert(0, ad)
                cnt[ad] = cnt.get(ad, 0) + k
                if cnt[ad] > 0 and ad not in on: on[ad] = ts
                if cnt[ad] == 0 and ad in on: spans.append((ad, on.pop(ad), ts))
                maxset = max(maxset, sum(1 for c in cnt.values() if c > 0))
            ads_used = sorted(set(g.adapter)); print(f"  [{a}] pod {p}: {len(g)} requests, {len(ads_used)} adapters; max running adapters {maxset} (slots {slots}); est. slot loads {loads} = CPU->GPU copies {copies} + disk reads {disk}")
            for ad, s0, s1 in spans: ax.broken_barh([((s0 - t0) / 1e6, (s1 - s0) / 1e6)], (order.index(ad) - 0.4, 0.8), color=plt.cm.tab20(order.index(ad) % 20))
            ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6); ax.set_ylim(len(order) - 0.5, -0.5)
            ax2 = ax.twinx(); st = D[a]["st"]; st = st[st.pod == p].sort_values("t_us"); ax2.plot((st.t_us - t0) / 1e6, st.deferred, "k-", lw=1.2, label="deferred (server log)"); ax2.set_ylabel("deferred"); ax2.legend(loc="upper right", fontsize=7)
            ax.set_title(f"{a} / pod {p}: adapters with a running request (bars) and deferred queue (line); {len(g)} requests, est. {copies} copies + {disk} disk reads", fontsize=9)
        np.atleast_1d(axes)[-1].set_xlabel("seconds since the arm's first send"); save(fig, f"residency_timeline_{a}.png")
for p in saved: print("Saved:", p)
