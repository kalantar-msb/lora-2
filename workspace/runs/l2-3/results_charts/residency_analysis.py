import glob, json, os, re, sys
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "l2-3"
ROOT = "/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2"
BASE = f"{ROOT}/workspace/runs/{RUN}/results"
OUT = f"{ROOT}/workspace/runs/{RUN}/results_charts"
os.makedirs(OUT, exist_ok=True)
arms = sorted(d for d in os.listdir(BASE) if os.path.isdir(f"{BASE}/{d}"))
arms = ["baseline"] + [a for a in arms if a != "baseline"]
WL = "granite-9adapter.workload"
pct = lambda s, q: float(np.percentile(s, q)) if len(s) else float("nan")

def cell_dirs(arm):
    return sorted(glob.glob(f"{BASE}/{arm}/{WL}/i*/")) or [f"{BASE}/{arm}/{WL}/"]

def load(arm):
    dirs = cell_dirs(arm)
    if len(dirs) > 1: print(f"note: {arm} has {len(dirs)} iterations, analysing all")
    d = dirs[0]
    tr = pd.read_csv(f"{d}/trace_data.csv"); tr = tr[tr.status == "ok"].copy()
    tr["ttft_ms"] = (tr.first_chunk_time_us - tr.send_time_us) / 1e3
    tr["e2e_ms"] = (tr.last_chunk_time_us - tr.send_time_us) / 1e3
    tr["tpot_ms"] = np.where(tr.output_tokens > 1, (tr.last_chunk_time_us - tr.first_chunk_time_us) / (tr.output_tokens - 1) / 1e3, np.nan)
    # pod IP -> short name
    ipmap = {}
    for f in glob.glob(f"{d}/resources/pod-*decode*.yaml"):
        t = open(f).read()
        ip = re.search(r"podIP: (\S+)", t); nm = re.search(r"^  name: (\S+)", t, re.M)
        if ip and nm: ipmap[ip.group(1)] = nm.group(1).rsplit("-", 1)[-1]
    # EPP per-request decisions
    recs, rem = [], {}
    for f in sorted(glob.glob(f"{d}/epp_logs/*.log")):
        for line in open(f, errors="replace"):
            if '"Completed running filter plugins"' in line:
                try: j = json.loads(line); rem[j.get("x-request-id")] = j.get("remainingEndpoints")
                except Exception: pass
            elif '"Request handled"' in line:
                try: j = json.loads(line)
                except Exception: continue
                recs.append(dict(t_us=int(pd.Timestamp(j["timestamp"]).value // 1000), rid=j.get("x-request-id"),
                                 adapter=j.get("targetModel"), pod=ipmap.get(j.get("endpoint", "").split(":")[0], j.get("endpoint"))))
    ep = pd.DataFrame(recs); ep["remaining"] = ep.rid.map(rem)
    # join trace rows to EPP decisions: same adapter, nearest time (EPP receive ~ client send)
    tr["send_time_us"] = tr.send_time_us.astype("int64"); tr = tr.sort_values("send_time_us"); ep = ep.sort_values("t_us")
    j = pd.merge_asof(tr, ep.rename(columns={"t_us": "epp_t_us"}), left_on="send_time_us", right_on="epp_t_us",
                      left_by="adapter", right_by="adapter", direction="nearest", tolerance=200_000)
    lag = (j.epp_t_us - j.send_time_us).abs() / 1e3
    print(f"[{arm}] join: {j.pod.notna().sum()}/{len(j)} trace rows matched to an EPP decision; "
          f"duplicate EPP matches={j.rid.duplicated().sum()}; |EPP-send| p50={lag.median():.1f} ms p99={pct(lag.dropna(),99):.1f} ms")
    # metrics scrapes
    mrows, lora = [], []
    for f in sorted(glob.glob(f"{d}/metrics/raw/*decode*_metrics.log")):
        pod = re.search(r"decode-[^-]+-([^_]+)_", f).group(1); ts = int(re.search(r"_(\d+)_metrics", f).group(1)) * 1e6
        run = wait = defr = np.nan
        for line in open(f, errors="replace"):
            if line.startswith("vllm:num_requests_running{"): run = float(line.split()[-1])
            elif line.startswith("vllm:num_requests_waiting{"): wait = float(line.split()[-1])
            elif line.startswith("vllm:num_requests_waiting_by_reason") and 'reason="deferred"' in line: defr = float(line.split()[-1])
            elif line.startswith("vllm:lora_requests_info{"):
                m = re.search(r'running_lora_adapters="([^"]*)"', line)
                lora.append(dict(pod=pod, scrape_us=ts, seen_us=float(line.split()[-1]) * 1e6, running=m.group(1)))
        mrows.append(dict(pod=pod, t_us=ts, running=run, waiting=wait, deferred=defr))
    met = pd.DataFrame(mrows); lora = pd.DataFrame(lora)
    # server-log stat lines (10 s)
    srows = []
    for f in glob.glob(f"{d}/server_logs/*.log"):
        pod = re.search(r"decode-[^-]+-([^_]+)_", f).group(1)
        for line in open(f, errors="replace"):
            m = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers.py.*Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?", line)
            if m:
                t = pd.Timestamp(f"2026-{m.group(1)}", tz="UTC").value / 1e3
                srows.append(dict(pod=pod, t_us=t, running=int(m.group(2)), waiting=int(m.group(3)), deferred=int(m.group(4) or 0)))
    st = pd.DataFrame(srows)
    maxl = sorted(set(re.findall(r'max_lora="(\d+)"', " ".join(open(f, errors="replace").read() for f in glob.glob(f"{d}/metrics/raw/*decode*_metrics.log")[:5]))))
    return dict(tr=j, ep=ep, met=met, lora=lora, st=st, ipmap=ipmap, max_lora=int(maxl[-1]) if maxl else None)

data = {a: load(a) for a in arms}
T0s = {a: d["tr"].send_time_us.min() for a, d in data.items()}

# ---------------- 1. latency table ----------------
print("\n=== Latency (status=ok; ms) ===")
rows = []
for a, d in data.items():
    t = d["tr"]
    for m in ("ttft_ms", "tpot_ms", "e2e_ms"):
        s = t[m].dropna(); rows.append(dict(arm=a, metric=m[:-3].upper(), n=len(s), mean=s.mean(), p50=pct(s,50), p90=pct(s,90), p99=pct(s,99)))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

print("\n=== Per-adapter TTFT p50 / p99 (ms) and routing (requests per pod) ===")
for a, d in data.items():
    t = d["tr"]; mat = t.pivot_table(index="adapter", columns="pod", values="request_id", aggfunc="count", fill_value=0)
    lat = t.groupby("adapter").ttft_ms.agg(n="count", p50=lambda s: pct(s,50), p99=lambda s: pct(s,99))
    tab = lat.join(mat).sort_values("n", ascending=False)
    print(f"\n[{a}]  pod slots (vLLM max_lora) = {d['max_lora']}")
    print(tab.to_string(float_format=lambda x: f"{x:,.0f}"))
    print(f"  per pod: " + ", ".join(f"{p}: n={len(g)} TTFT p50={pct(g.ttft_ms,50):,.0f} p99={pct(g.ttft_ms,99):,.0f}" for p, g in t.groupby("pod")))
    rr = t.remaining.value_counts().sort_index(); print(f"  candidate pods left after filters (adapter requests): " + ", ".join(f"{int(k)} pod(s): {v}" for k, v in rr.items()))

# ---------------- 2. residency reconstruction ----------------
def reconstruct(t, pod, slots):
    g = t[t.pod == pod]
    ev = []
    for r in g.itertuples():
        ev += [(r.first_chunk_time_us, 1, r.adapter), (r.last_chunk_time_us, -1, r.adapter)]
    ev.sort(key=lambda x: (x[0], x[1]))
    cnt, run_set_log, lru, loads, loads_by = {}, [], [], 0, {}
    for ts, k, ad in ev:
        if k == 1:
            if ad not in lru:
                loads += 1; loads_by[ad] = loads_by.get(ad, 0) + 1
                if len(lru) >= slots: lru.pop(0)
                lru.append(ad)
            else:
                lru.remove(ad); lru.append(ad)
        cnt[ad] = cnt.get(ad, 0) + k
        run_set_log.append((ts, frozenset(a for a, c in cnt.items() if c > 0)))
    waiting = [(r.send_time_us, r.first_chunk_time_us) for r in g.itertuples()]
    return run_set_log, loads, loads_by, waiting

print("\n=== Reconstructed residency per pod (requests' first-token..last-token intervals; LRU slot replay) ===")
recon = {}
for a, d in data.items():
    slots = d["max_lora"] or 2
    for pod in sorted(d["tr"].pod.dropna().unique()):
        log, loads, loads_by, waiting = reconstruct(d["tr"], pod, slots)
        recon[(a, pod)] = (log, waiting)
        maxset = max(len(s) for _, s in log)
        ads = sorted(set().union(*[s for _, s in log]))
        # validate against vLLM's own counters at scrape times
        met = d["met"][d["met"].pod == pod].sort_values("t_us")
        g = d["tr"][d["tr"].pod == pod]
        rc_run = [((g.first_chunk_time_us <= ts) & (g.last_chunk_time_us > ts)).sum() for ts in met.t_us]
        rc_wait = [((g.send_time_us <= ts) & (g.first_chunk_time_us > ts)).sum() for ts in met.t_us]
        dr = np.abs(np.array(rc_run) - met.running.values); dw = np.abs(np.array(rc_wait) - met.deferred.values)
        lr = d["lora"][d["lora"].pod == pod]
        obs_sets = {frozenset(x for x in s.split(",") if x) for s in lr.running}
        rec_sets = {s for _, s in log}
        print(f"[{a}] pod {pod}: adapters run {len(ads)} {ads}; max simultaneous running adapters {maxset} (slots {slots}); "
              f"estimated slot loads {loads}  {dict(sorted(loads_by.items(), key=lambda x:-x[1]))}")
        print(f"      check vs vLLM metrics at {len(met)} scrapes: running |diff| median {np.median(dr):.0f} max {dr.max():.0f}; "
              f"waiting-vs-deferred |diff| median {np.median(dw):.0f} max {dw.max():.0f}; "
              f"vLLM running-adapter sets observed {len(obs_sets)}, all within reconstructed sets: {obs_sets <= rec_sets | {frozenset()}}")

# ---------------- charts ----------------
cols = plt.get_cmap("tab10")
all_ads = data["baseline"]["tr"].adapter.value_counts().index.tolist()
color = {ad: cols(i % 10) for i, ad in enumerate(all_ads)}
pods = [(a, p) for (a, p) in recon]
fig, axes = plt.subplots(len(pods), 1, figsize=(13, 3.4 * len(pods)), sharex=True)
for ax, (a, p) in zip(axes, pods):
    log, waiting = recon[(a, p)]
    ts = np.array([x for x, _ in log]); 
    for i, ad in enumerate(all_ads):
        on = np.array([ad in s for _, s in log])
        start = None
        for k in range(len(ts)):
            if on[k] and start is None: start = ts[k]
            if (not on[k]) and start is not None:
                ax.broken_barh([((start - T0s[a]) / 1e6, (ts[k] - start) / 1e6)], (i - 0.4, 0.8), color=color[ad]); start = None
    ax.set_yticks(range(len(all_ads))); ax.set_yticklabels(all_ads, fontsize=7)
    ax2 = ax.twinx()
    st = data[a]["st"]; st = st[st.pod == p].sort_values("t_us")
    ax2.plot((st.t_us - T0s[a]) / 1e6, st.deferred, "k-", lw=1.5, label="vLLM deferred (server log, 10 s)")
    met = data[a]["met"]; met = met[met.pod == p].sort_values("t_us")
    ax2.plot((met.t_us - T0s[a]) / 1e6, met.deferred, "k:", lw=1, label="deferred gauge (metrics, 15 s)")
    ax2.set_ylabel("requests deferred"); ax2.legend(loc="upper right", fontsize=7)
    n = (data[a]["tr"].pod == p).sum()
    ax.set_title(f"{a} / pod {p}: adapters with a running request (bars) and deferred queue (line); {n} requests", fontsize=9)
axes[-1].set_xlabel("seconds since the arm's first send"); axes[-1].set_xlim(-2, 210)
plt.tight_layout(); f1 = f"{OUT}/residency_timeline.png"; plt.savefig(f1, dpi=110); plt.close()

fig, axes = plt.subplots(1, len(arms), figsize=(14, 5), sharey=True)
for ax, a in zip(axes, arms):
    t = data[a]["tr"]
    for ad in all_ads:
        g = t[t.adapter == ad]; ax.scatter((g.send_time_us - T0s[a]) / 1e6, g.ttft_ms / 1e3, s=6, color=color[ad], label=f"{ad} ({len(g)})")
    ax.set_yscale("log"); ax.set_title(f"{a}: TTFT by send time"); ax.set_xlabel("seconds since first send")
axes[0].set_ylabel("TTFT (s, log)"); axes[-1].legend(fontsize=7, loc="lower right")
plt.tight_layout(); f2 = f"{OUT}/ttft_by_send_time.png"; plt.savefig(f2, dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for a in arms:
    s = np.sort(data[a]["tr"].ttft_ms.values / 1e3); ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", label=a)
ax.set_xscale("log"); ax.set_xlabel("TTFT (s, log)"); ax.set_ylabel("fraction of requests"); ax.legend(); ax.set_title("TTFT CDF")
plt.tight_layout(); f3 = f"{OUT}/ttft_cdf.png"; plt.savefig(f3, dpi=110); plt.close()
for f in (f1, f2, f3): print(f"Saved: {f}")
