import glob, json, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = "l2-1"
base = f"workspace/runs/{run}/results"
out = f"workspace/runs/{run}/results_charts"
os.makedirs(out, exist_ok=True)
WL = "granite-9adapter.workload"
arms = ["baseline"] + sorted(a for a in os.listdir(base)
                            if a not in ("baseline",) and os.path.isdir(f"{base}/{a}"))
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

def cells(arm):
    return sorted(glob.glob(f"{base}/{arm}/{WL}/*/trace_data.csv"))

def epp_routes(cell):
    m = {}
    for f in glob.glob(f"{cell}/epp_logs/*.log"):
        for line in open(f):
            if '"Request handled"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            m.setdefault("rows", []).append((pd.Timestamp(d["timestamp"]).value // 1000,
                                             d.get("targetModel"), d.get("endpoint", "").split(":")[0]))
    return pd.DataFrame(m.get("rows", []), columns=["epp_us", "adapter", "pod_ip"]).sort_values("epp_us")

def pod_ips(cell):
    ips = {}
    for f in glob.glob(f"{cell}/resources/**/*", recursive=True):
        if os.path.isfile(f):
            txt = open(f, errors="ignore").read()
            for pod, ip in re.findall(r"name: (ibm-gran-\S+-decode-\S+)\n(?:.*\n)*?\s+podIP: (\S+)", txt):
                ips[ip] = pod
    return ips

def metrics(cell):
    rows = []
    for f in glob.glob(f"{cell}/metrics/raw/*decode*_metrics.log"):
        mm = re.search(r"/(ibm-gran-\S+?)_(\d+)_metrics\.log$", f)
        pod, ts = mm.group(1), int(mm.group(2))
        for line in open(f):
            if line.startswith("#"):
                continue
            for name in ("vllm:num_requests_waiting", "vllm:num_requests_running",
                         "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc"):
                if line.startswith(name + "{") or line.startswith(name + " "):
                    rows.append((pod, ts, name.split(":")[1], float(line.split()[-1])))
            if line.startswith("vllm:lora_requests_info{"):
                r = re.search(r'running_lora_adapters="([^"]*)"', line).group(1)
                w = re.search(r'waiting_lora_adapters="([^"]*)"', line).group(1)
                rows.append((pod, ts, "lora_running_n", len([x for x in r.split(",") if x])))
                rows.append((pod, ts, "lora_waiting_n", len([x for x in w.split(",") if x])))
    return pd.DataFrame(rows, columns=["pod", "ts", "metric", "value"])

def pct(s, q):
    return s.quantile(q) if len(s) else float("nan")

frames, mets, loads = {}, {}, {}
for arm in arms:
    parts = []
    for tf in cells(arm):
        cell = os.path.dirname(tf)
        df = pd.read_csv(tf)
        routes = epp_routes(cell)
        ips = pod_ips(cell)
        # The trace's x_request_id is replaced upstream of the EPP, so match each request
        # to the next EPP decision for the same adapter within 2 s of its send time.
        df = pd.merge_asof(df.sort_values("send_time_us"), routes.rename(columns={"adapter": "adapter"}),
                           left_on="send_time_us", right_on="epp_us", by="adapter",
                           direction="forward", tolerance=2_000_000)
        lag = (df.epp_us - df.send_time_us) / 1000
        print(f"{arm}: matched {df.pod_ip.notna().sum()}/{len(df)}; send->EPP lag ms p50={lag.median():.1f} p99={lag.quantile(.99):.1f}; "
              f"duplicate EPP matches={df.epp_us.duplicated(keep=False).sum() - df.epp_us.isna().sum()}")
        df["pod"] = df["pod_ip"].map(lambda ip: ips.get(ip, ip))
        df["iter"] = os.path.basename(cell)
        parts.append(df)
        mets[arm] = metrics(cell)
        loads[arm] = {os.path.basename(s).replace("_vllm.log", ""):
                      sum(1 for l in open(s) if "Resolved and loaded LoRA adapter" in l)
                      for s in glob.glob(f"{cell}/server_logs/*_vllm.log")}
    if len(parts) > 1:
        print(f"note: {arm} has {len(parts)} iterations — pooled")
    df = pd.concat(parts, ignore_index=True)
    df["ttft_ms"] = (df.first_chunk_time_us - df.send_time_us) / 1000
    df["e2e_ms"] = (df.last_chunk_time_us - df.send_time_us) / 1000
    df["tpot_ms"] = ((df.last_chunk_time_us - df.first_chunk_time_us)
                     / (df.output_tokens - 1)).where(df.output_tokens > 1) / 1000
    frames[arm] = df

# ── Global ─────────────────────────────────────────────────────────────
print("\n=== GLOBAL ===")
g = []
for arm, df in frames.items():
    ok = df[df.status == "ok"]
    dur = (ok.last_chunk_time_us.max() - ok.send_time_us.min()) / 1e6
    send_span = (ok.send_time_us.max() - ok.send_time_us.min()) / 1e6
    g.append({
        "arm": arm, "requests": len(df), "ok": len(ok),
        "unrouted": int(df.pod.isna().sum()),
        "send_rate_rps": len(ok) / send_span,
        "done_rate_rps": len(ok) / dur, "wall_s": dur,
        "out_tok_per_s": ok.output_tokens.sum() / dur,
        "ttft_p50": pct(ok.ttft_ms, .5), "ttft_mean": ok.ttft_ms.mean(),
        "ttft_p90": pct(ok.ttft_ms, .9), "ttft_p99": pct(ok.ttft_ms, .99),
        "e2e_p50": pct(ok.e2e_ms, .5), "e2e_p99": pct(ok.e2e_ms, .99),
        "tpot_p50": pct(ok.tpot_ms, .5), "tpot_p99": pct(ok.tpot_ms, .99),
    })
print(pd.DataFrame(g).set_index("arm").T.round(2).to_string())

# Paired, same request_id in both arms
b, t = frames["baseline"], frames[arms[1]]
pair = b[b.status == "ok"].merge(t[t.status == "ok"], on="request_id", suffixes=("_b", "_t"))
print(f"\npaired requests: {len(pair)}")
print(f"  {arms[1]} TTFT lower than baseline on {(pair.ttft_ms_t < pair.ttft_ms_b).mean():.1%} of requests")
print(f"  median TTFT difference ({arms[1]} - baseline): {(pair.ttft_ms_t - pair.ttft_ms_b).median():.0f} ms")

# Server-side
print("\n=== SERVER SIDE (decode pods, 15 s scrapes) ===")
srv = []
for arm in arms:
    m = mets[arm]
    for pod, pm in m.groupby("pod"):
        def s(name):
            return pm[pm.metric == name].sort_values("ts").value
        pre = s("num_preemptions_total")
        srv.append({"arm": arm, "pod": pod[-5:],
                    "requests_routed": int((frames[arm].pod == pod).sum()),
                    "waiting_max": s("num_requests_waiting").max(),
                    "waiting_mean": s("num_requests_waiting").mean(),
                    "running_max": s("num_requests_running").max(),
                    "kv_max": s("kv_cache_usage_perc").max(),
                    "preemptions": pre.iloc[-1] - pre.iloc[0] if len(pre) else float("nan"),
                    "lora_waiting_max": s("lora_waiting_n").max(),
                    "disk_loads": loads[arm].get(pod, float("nan"))})
print(pd.DataFrame(srv).round(2).to_string(index=False))

# ── Per adapter ────────────────────────────────────────────────────────
print("\n=== PER ADAPTER ===")
rows = []
for arm, df in frames.items():
    ok = df[df.status == "ok"]
    pods = sorted(ok.pod.dropna().unique())
    for ad, a in ok.groupby("adapter"):
        share = a.pod.value_counts(normalize=True)
        rows.append({"adapter": ad, "arm": arm, "n": len(a),
                     "ttft_p50": pct(a.ttft_ms, .5), "ttft_p90": pct(a.ttft_ms, .9),
                     "ttft_p99": pct(a.ttft_ms, .99),
                     "e2e_p50": pct(a.e2e_ms, .5), "tpot_p50": pct(a.tpot_ms, .5),
                     "pods_used": a.pod.nunique(),
                     "top_pod_share": share.max(),
                     **{f"on_{p[-5:]}": int((a.pod == p).sum()) for p in pods}})
pa = pd.DataFrame(rows)
order = frames["baseline"].adapter.value_counts().index
pa["adapter"] = pd.Categorical(pa.adapter, order, ordered=True)
pa = pa.sort_values(["adapter", "arm"])
cols = ["adapter", "arm", "n", "ttft_p50", "ttft_p90", "ttft_p99", "e2e_p50", "tpot_p50",
        "pods_used", "top_pod_share"]
print(pa[cols].round(2).to_string(index=False))
print("\nplacement (requests per pod):")
print(pa[["adapter", "arm"] + [c for c in pa.columns if c.startswith("on_")]].fillna("").to_string(index=False))

# Paired per adapter
pp = pair.groupby("adapter_b").apply(lambda x: pd.Series({
    "n": len(x),
    "faster_share": (x.ttft_ms_t < x.ttft_ms_b).mean(),
    "median_ttft_diff_ms": (x.ttft_ms_t - x.ttft_ms_b).median()}), include_groups=False)
print(f"\npaired per adapter ({arms[1]} vs baseline):")
print(pp.reindex(order).round(2).to_string())

# ── Charts ─────────────────────────────────────────────────────────────
colors = {"baseline": "#6b7280", arms[1]: "#2563eb"}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for arm, df in frames.items():
    ok = df[df.status == "ok"]
    for i, col in enumerate(["ttft_ms", "e2e_ms"]):
        v = ok[col].sort_values().values / 1000
        ax[i].plot(v, [j / len(v) for j in range(len(v))], label=arm, color=colors[arm])
for i, t_ in enumerate(["TTFT", "E2E latency"]):
    ax[i].set_xscale("log"); ax[i].set_xlabel(f"{t_} (s, log)"); ax[i].set_ylabel("fraction of requests")
    ax[i].set_title(f"{t_} CDF — all adapters"); ax[i].grid(alpha=.3); ax[i].legend()
plt.tight_layout(); plt.savefig(f"{out}/global_latency_cdf.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
x = range(len(order)); w = 0.4
for k, q in enumerate(["ttft_p50", "ttft_p99"]):
    for j, arm in enumerate(arms):
        v = pa[pa.arm == arm].set_index("adapter").reindex(order)[q] / 1000
        ax[k].bar([i + (j - .5) * w for i in x], v, w, label=arm, color=colors[arm])
    ax[k].set_xticks(list(x)); ax[k].set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax[k].set_ylabel("seconds"); ax[k].set_title(f"{q.replace('_', ' ').upper()} by adapter (most → least traffic)")
    ax[k].legend(); ax[k].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(f"{out}/per_adapter_ttft.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, len(arms), figsize=(13, 4.5), sharey=True)
for k, arm in enumerate(arms):
    ok = frames[arm][frames[arm].status == "ok"]
    ct = pd.crosstab(ok.adapter, ok.pod).reindex(order)
    ct.columns = [c[-5:] for c in ct.columns]
    ct.plot.barh(stacked=True, ax=ax[k], color=["#f59e0b", "#10b981"])
    ax[k].set_title(f"{arm}: requests per pod"); ax[k].set_xlabel("requests"); ax[k].invert_yaxis()
plt.tight_layout(); plt.savefig(f"{out}/per_adapter_placement.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, len(arms), figsize=(13, 4), sharey=True)
for k, arm in enumerate(arms):
    m = mets[arm]; w_ = m[m.metric == "num_requests_waiting"]
    t0 = w_.ts.min()
    for pod, pm in w_.groupby("pod"):
        pm = pm.sort_values("ts"); ax[k].plot(pm.ts - t0, pm.value, marker=".", label=pod[-5:])
    ax[k].set_title(f"{arm}: requests waiting per pod"); ax[k].set_xlabel("seconds"); ax[k].legend(); ax[k].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{out}/queue_depth.png", dpi=120); plt.close()

for f in ["global_latency_cdf", "per_adapter_ttft", "per_adapter_placement", "queue_depth"]:
    print(f"Saved: {out}/{f}.png")
