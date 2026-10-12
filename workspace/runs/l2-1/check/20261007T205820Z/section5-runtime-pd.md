# 5. Runtime Analysis — run l2-1

**Cells (PRESENT rows):** `baseline | granite-9adapter.workload | i1` and `routetoholder | granite-9adapter.workload | i1`
**Results:** `$RESULTS_DIR = /Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2/workspace/runs/l2-1/results`
**Analysis scripts (read-only, `python3 -I`):** `/tmp/s2r-check-scripts/a3/{thermal,adapters,lora,epp_bodies,flags,join,timing}.py`

**Operator-acknowledged limitations (not failed on):**
1. No simulation exists for this cell (granite-3.3-2b, 9 adapters, 2 pods, 3 slots). "Expected" values come from `workloads/granite-9adapter.workload.yaml`, `config.md` and the specification — latency/TTFT parity vs sim is INAPPLICABLE; TTFT below is descriptive only.
2. The measuring BLIS tree and the algorithm-source BLIS tree differ.
3. Baseline ran on EPP `...-5b9c84f9ff-b9r66` (no `--allow-experimental-plugins`), routetoholder on EPP `...-666d8b6ff6-xkdt7` (flag set). Verified in 5c: that flag is the only EPP flag difference.

**How request lines name the adapter (needed for 5b.1).** vLLM access lines carry no model name and no timestamp (`"POST /v1/completions HTTP/1.1" 200 OK`), so vLLM logs give per-pod totals only. The adapter appears on the wire as the request `model`, which the EPP logs in every `"Request handled"` line (`incomingModelName`, plus `endpoint` = pod IP). The per-adapter × per-pod matrix is therefore taken from EPP routing decisions and cross-checked against vLLM: per-pod EPP totals equal vLLM `POST /v1/completions` counts and `vllm:request_success_total` exactly (535/635 baseline, 689/495 routetoholder).

**Joining trace rows to routing decisions.** `trace_data.csv:x_request_id` matches **0/950** EPP `x-request-id` values in both arms (the id the EPP logs is not the one the client recorded). Rows were joined per adapter in time order: the last *n_a* EPP requests for adapter *a* ↔ the *n_a* trace rows for *a*, sorted by `send_time_us` (the trace is the tail of the run; the 9 preflight + 50 warmup requests precede it). Check: EPP-handled minus client-send is −0.2 ms … +20 ms (p50 0.2 ms baseline / 3.1 ms routetoholder; the 105 tiny negatives in baseline are sub-millisecond clock offset). Per-pod totals from the join reconcile with vLLM counts (see 5b).

**Traffic that is not in the trace** (all of it ends before the first trace send):

| arm | base-model requests (`ibm-granite/granite-3.3-2b-instruct`) | adapter preflight (1 per adapter) | warmup (excluded from trace) | trace rows | EPP total = vLLM total |
|---|---|---|---|---|---|
| baseline | 161 (19:46:16–19:47:29) | 9 | 50 | 950 | 1170 |
| routetoholder | 175 (20:14:48–20:16:10) | 9 | 50 | 950 | 1184 |

What sends the base-model requests is not identified from these artifacts.

---

## 5h. GPU Thermal/Power Health (run first)

**What:** temperature and power of each pod's GPU during the workload window, to check for throttling.
**Expected:** H100 SXM5 thresholds (spec §5h); need ≥ 30 samples with `util == 100`.
**Actual:** pod → UUID (`gpu_logs/<pod>.uuids`) → host index (`<node>.gpus.csv`) → `<node>.log` samples. Window = min(`send_time_us`) … max(`last_chunk_time_us`) of ok rows: baseline 19:47:36.9–19:50:12.5 (155.6 s); routetoholder 20:16:17.4–20:18:35.4 (137.9 s). Samples are ~12 s apart.

| phase | workload | pod | node | gpu | samples in window | util==100 | util p50 / max | temp_p95 | temp_max | pwr_p10 | cap_min | throttle% | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | granite-9adapter | 9q5qd | pokprod-b93r38s1 | 1 | 13 | 0 | 16% / 57% | 35.4 °C | 36 °C | 169 W | 700 W | n/a | **SKIP** (insufficient steady-state samples) |
| baseline | granite-9adapter | qwncq | pokprod-b93r44s2 | 4 | 13 | 0 | 17% / 18% | 32.0 °C | 32 °C | 167 W | 700 W | n/a | **SKIP** (insufficient steady-state samples) |
| routetoholder | granite-9adapter | 2pxst | pokprod-b93r44s2 | 5 | 12 | 0 | 18% / 21% | 30.0 °C | 30 °C | 148 W | 700 W | n/a | **SKIP** (insufficient steady-state samples) |
| routetoholder | granite-9adapter | x8rtb | pokprod-b93r43s1 | 1 | 12 | 0 | 17% / 20% | 34.0 °C | 34 °C | 176 W | 700 W | n/a | **SKIP** (insufficient steady-state samples) |

(temp/power columns above are over all in-window samples, shown for context; the spec's metrics are defined on `util==100` samples, of which there are none.)

**Why SKIP:** a 2B model at 10 req/s never drives the GPU to 100% utilisation, so the spec's steady-state filter leaves zero samples. Throttle status is formally unknown. Context only: the hottest GPU reached 36 °C and every power cap is 700 W (the H100 SXM5 design TDP), so nothing points to throttling.

- Rollup: 0 PASS / 0 WARN / 0 FAIL / 2 SKIP cells. **No cell is tagged `thermal-FAIL`.**
- Persistent hotspots: none.
- Note: the two arms ran on different GPUs (baseline r38s1/1 + r44s2/4; routetoholder r44s2/5 + r43s1/1). All are "NVIDIA H100 80GB HBM3".
- Inferential limit: the table-format `nvidia-smi` capture has no `clocks_throttle_reasons.*` or clock fields, so throttling can only be inferred from temperature/power.

---

## 5a. vLLM Config Validation

**What:** the `non-default args` and engine-init lines of each vLLM log vs `config.md` §2–3.
**Actual:** all four logs carry an identical `non-default args` dict (`server_logs/*_vllm.log`, `api_utils.py:286`), vLLM v0.30.0 (`docker.io/vllm/vllm-openai:v0.30.0`).

| field | expected (config.md) | baseline 9q5qd | baseline qwncq | rth 2pxst | rth x8rtb | verdict |
|---|---|---|---|---|---|---|
| model | ibm-granite/granite-3.3-2b-instruct | same | same | same | same | PASS |
| tensor_parallel_size | 1 | 1 | 1 | 1 | 1 | PASS |
| max_model_len | 8192 | 8192 | 8192 | 8192 | 8192 | PASS |
| gpu_memory_utilization | 0.90 | 0.9 | 0.9 | 0.9 | 0.9 | PASS |
| enable_prefix_caching | False | False | False | False | False | PASS |
| max_num_seqs | 64 | 64 | 64 | 64 | 64 | PASS |
| max_num_batched_tokens | (not set) | 8192 (chunked-prefill default) | 8192 | 8192 | 8192 | PASS (equal) |
| block_size / dtype | 16 / bfloat16 | 16 / bfloat16 | same | same | same | PASS |
| enable_lora / max_loras / max_lora_rank / max_cpu_loras | set / 3 / 32 / 6 | True / 3 / 32 / 6 | same | same | same | PASS |
| VLLM_PLUGINS / VLLM_LORA_RESOLVER_CACHE_DIR / VLLM_ALLOW_RUNTIME_LORA_UPDATING (pod env) | lora_filesystem_resolver / /adapters / true | same | same | same | same | PASS |

Runtime confirmation of the resolver: every pod logs `Resolved and loaded LoRA adapter '<name>' using FilesystemResolver` (`serving.py:313`). `vllm:lora_requests_info{max_lora="3"}` on every pod.

**Verdict: PASS (both cells).**

---
## 5b. Instance Health & Request Distribution

**What:** requests per vLLM pod, total served vs sent, and whether the split matches each arm's routing policy.
**Expected:**
- baseline — load-aware scorers (queue 2, kv-util 2, prefix 3, max-score-picker): roughly even, spread < 20% (WARN > 20%, FAIL > 40%).
- routetoholder — affinity filter (spec `algorithms/routetoholder.go`; config.md §8): each adapter should go to the pod the EPP believes holds it, so traffic is skewed **by adapter**, and the skew should line up with adapter residency rather than random hot-spotting. The request-count split then follows whichever adapters each pod holds.

**Actual — per-pod counts.** "vLLM POST" = `grep -c 'POST /v1/completions'` per server log (the whole log, including pre-trace traffic). "trace window" = trace rows joined to EPP decisions (see header).

| phase/workload | pod | vLLM POST (all) | EPP-routed (all) | vLLM success_total | trace-window reqs | trace-window % | spread (trace window) | spread (all) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| baseline/granite-9adapter | 9q5qd | 535 | 535 | 535 | 429 | 45.2% | 19.4% | 17.1% | **PASS** (just under 20%) |
| baseline/granite-9adapter | qwncq | 635 | 635 | 635 | 521 | 54.8% | | | |
| routetoholder/granite-9adapter | 2pxst | 689 | 689 | 689 | 483 | 50.8% | 3.4% | 32.8% | **PASS** (affinity; see below) |
| routetoholder/granite-9adapter | x8rtb | 495 | 495 | 495 | 467 | 49.2% | | | |

- Total served = total sent: 950/950 trace rows in each arm reached a pod (no shedding; 5e). vLLM totals 1170 / 1184 = EPP `Request handled` totals.
- Health: both pods in each arm served traffic, `restartCount: 0` for every vLLM, sidecar and EPP container (`resources/pod-*.yaml`), 0 ERROR lines (5f), 0 non-200 responses.
- routetoholder's "all" spread (32.8%) comes from pre-trace base-model requests (173 of 175 went to 2pxst). Base-model requests skip the filter (5c), so this is the stock scorers' choice, not the treatment.
- **Routetoholder skew lines up with residency:** every adapter went to exactly one pod (table 5b.1-B) and the filter reduced the candidates to 1 pod for 950/950 trace requests (5c). Expected request share by adapter set: 2pxst's adapters {answerability, uncertainty, hallucination_detection} = 54.2% of rate_fraction, x8rtb's six = 45.8%. Observed: 50.8% / 49.2%.
- Cross-phase: the two arms split traffic differently. That is expected here — the only routing-config difference is the filter (5c) — and it is not a deployment fault.
- No `thermal-FAIL` cells (5h).

**Verdict:** baseline **PASS**; routetoholder **PASS** for distribution (the skew follows residency). The concern about *which* adapters were grouped together is raised in 5b.1-D.

### 5b.1 Adapter Traffic Coverage & Proportion (server side)

#### A. Coverage — did every adapter reach a pod?

**What:** every one of the 9 adapters with non-zero `rate_fraction` reaches at least one pod in each arm.
**Expected:** all 9 (`workloads/granite-9adapter.workload.yaml`, `clients[*].adapter`).
**Actual:** three independent server-side sources agree.

| source | baseline 9q5qd | baseline qwncq | rth 2pxst | rth x8rtb | union per arm |
|---|---|---|---|---|---|
| EPP `Request handled` (trace window), adapters with ≥ 1 request | 9 | 9 | 3 | 6 | 9 / 9 |
| vLLM `Resolved and loaded LoRA adapter` (first-use registration, whole log) | 9 | 9 | 5 (+answer_relevance_classifier, query_rewrite from 1 preflight each) | 6 | 9 / 9 |
| `vllm:lora_requests_info` `running_lora_adapters`, union over all scrapes | 9 | 9 | 3 | 6 | 9 / 9 |

**Verdict: PASS (both arms)** — all 9 adapters reached a pod and were run there.

#### B. Per-adapter × per-pod request matrix (trace window; in parentheses: all adapter requests including the 1 preflight + warmup)

| adapter | rate_fraction | baseline 9q5qd | baseline qwncq | baseline top-pod share | rth 2pxst | rth x8rtb | rth top-pod share |
|---|---|---|---|---|---|---|---|
| answerability | 35.3% | 138 (144) | 188 (197) | 58% | 326 (341) | 0 (0) | 100% |
| citations | 17.7% | 79 (83) | 85 (91) | 52% | 0 (0) | 164 (174) | 100% |
| uncertainty | 11.8% | 40 (44) | 56 (63) | 58% | 96 (107) | 0 (0) | 100% |
| context_relevance | 8.8% | 38 (40) | 54 (56) | 59% | 0 (0) | 92 (96) | 100% |
| hallucination_detection | 7.1% | 23 (26) | 38 (40) | 62% | 61 (66) | 0 (0) | 100% |
| query_rewrite | 5.9% | 28 (29) | 36 (38) | 56% | 0 (1) | 64 (66) | 100% |
| answer_relevance_classifier | 5.0% | 31 (32) | 27 (29) | 53% | 0 (1) | 58 (60) | 100% |
| requirement_check | 4.4% | 28 (31) | 27 (28) | 51% | 0 (0) | 55 (59) | 100% |
| answer_relevance_rewriter | 3.9% | 24 (28) | 10 (10) | 71% | 0 (0) | 34 (38) | 100% |
| **total** | | 429 | 521 | traffic-weighted 56.8% | 483 | 467 | traffic-weighted 100% |

- Baseline: every adapter is split across both pods (51–71% on the busier pod) → each pod needs all 9 adapters on 3 GPU slots.
- Routetoholder: strict partition. 2pxst serves 3 adapters (= its 3 slots); x8rtb serves 6 adapters (= 2× its 3 slots). The only cross-pod requests are one preflight each for answer_relevance_classifier and query_rewrite on 2pxst, before the warmup.

#### C. Proportion — do the counts match `rate_fraction`?

**What:** per-adapter counts vs the workload's Zipf(α=1.0) `rate_fraction`, judged statistically (950 requests is a multinomial sample).
**Expected:** `rate_fraction` normalised over the 9 clients (sum = 1.000). Expected *n* = share × 950; 3σ band = binomial ±3·√(N·p·(1−p)).
**Actual:** the trace's per-adapter counts are **identical in both arms** (same seed 42). The EPP-side trace-window counts equal the trace counts by construction of the join. The independent EPP check (all adapter requests, minus the one preflight each, = warmup + trace, N = 1000) is shown in the last column.

| adapter | expected share | expected n | trace n (both arms) | trace share | diff (pp) | z | 3σ band | in band | ok n / ok % | EPP n (warmup+trace, N=1000) |
|---|---|---|---|---|---|---|---|---|---|---|
| answerability | 35.35% | 335.8 | 326 | 34.32% | −1.03 | −0.67 | 292–380 | yes | 326 / 100% | 340 |
| citations | 17.67% | 167.9 | 164 | 17.26% | −0.41 | −0.33 | 133–203 | yes | 164 / 100% | 173 |
| uncertainty | 11.78% | 111.9 | 96 | 10.11% | −1.68 | −1.60 | 82–142 | yes | 96 / 100% | 106 |
| context_relevance | 8.84% | 84.0 | 92 | 9.68% | +0.85 | +0.92 | 58–110 | yes | 92 / 100% | 95 |
| hallucination_detection | 7.07% | 67.2 | 61 | 6.42% | −0.65 | −0.78 | 43–91 | yes | 61 / 100% | 65 |
| query_rewrite | 5.89% | 56.0 | 64 | 6.74% | +0.85 | +1.11 | 34–78 | yes | 64 / 100% | 66 |
| answer_relevance_classifier | 5.05% | 48.0 | 58 | 6.11% | +1.06 | +1.49 | 28–68 | yes | 58 / 100% | 60 |
| requirement_check | 4.42% | 42.0 | 55 | 5.79% | +1.37 | +2.06 | 23–61 | yes | 55 / 100% | 58 |
| answer_relevance_rewriter | 3.93% | 37.3 | 34 | 3.58% | −0.35 | −0.55 | 19–55 | yes | 34 / 100% | 37 |

- Trace (N=950): χ² = 11.57, df = 8, **p = 0.172**. Largest pp difference −1.68 (uncertainty); largest |z| = 2.06 (requirement_check). All 9 inside 3σ.
- EPP warmup+trace (N=1000): χ² = 9.83, df = 8, **p = 0.277**.
- Warmup: the first 50 dispatched requests are excluded from the trace (`trace_header.yaml: warm_up_requests: 50`). They plus the 9 preflights are the 59 "pre-trace" adapter requests, distributed {answerability 15, citations 10, uncertainty 11, context_relevance 4, hallucination_detection 5, query_rewrite 3, answer_relevance_classifier 3, requirement_check 4, answer_relevance_rewriter 4}.

**Verdict: PASS (both arms)** — counts match `rate_fraction` within sampling error; every adapter ran with 100% `status=ok`. This agrees with Agent 1's client-side check, since the trace counts are the same data.

#### D. Adapter loads / evictions per pod (the mechanism's primary outcome)

**What can be counted.** vLLM v0.30.0 at these log settings writes **no per-load or per-eviction line** for GPU slots. The only `unload` match is the startup warning `LoRA dynamic loading & unloading is enabled`. `Resolved and loaded LoRA adapter` fires once per adapter per pod (first use, registration with the resolver), not per GPU slot load. `vllm:lora_requests_info` keeps one series per distinct (running, waiting) adapter-set label, valued with the last time that set was seen. A set that recurs overwrites its timestamp, so the series cannot be replayed into a load count. **GPU adapter loads and evictions cannot be counted from these artifacts.** What can be measured is how many adapters each pod cycled through its 3 slots, and how many requests vLLM held back because their adapter had no free slot ("Deferred" in `loggers.py` stats, `vllm:num_requests_waiting_by_reason{reason="deferred"}`).

| arm | pod | adapters routed (trace) | distinct adapters ever in running set | distinct running/waiting sets | max running-set size | engine `Waiting` max / mean | `Deferred` max / mean (stat lines with Deferred > 0) | deferred gauge max (metrics) | KV usage max | preemptions |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 9q5qd | 9 | 9 | 28 | 3 | 142 / 61.4 | 142 / 61.4 (14 of 17) | 135 | 2.2% | 0 |
| baseline | qwncq | 9 | 9 | 26 | 3 | 139 / 62.1 | 139 / 62.1 (13 of 17) | 139 | 1.7% | 0 |
| routetoholder | 2pxst | 3 | 3 | 6 | 3 | **0 / 0.0** | **0 / 0.0 (0 of 15)** | **0** | 1.4% | 0 |
| routetoholder | x8rtb | 6 | 6 | 16 | 3 | 128 / 57.3 | 128 / 57.3 (13 of 15) | 125 | 2.3% | 0 |

- In every pod `Waiting` equals `Deferred`, and KV cache usage stays ≤ 2.3%. All queueing in this cell is LoRA-slot deferral, not memory pressure.
- Baseline: both pods cycle all 9 adapters through 3 slots, so both defer heavily.
- Routetoholder: 2pxst's working set (3 adapters) fits its 3 slots. After warm-up it never deferred a request and its running set never contained any other adapter, which points to no further slot swaps there. x8rtb's working set (6 adapters) is twice its slot count. It cycles 6 adapters and defers about as much as either baseline pod.
- **Finding (WARN, routetoholder):** for all 950 trace requests the filter left exactly 1 candidate (5c). For x8rtb's six adapters that candidate was always x8rtb, even while x8rtb had over 100 requests deferred and 2pxst had none. If no pod had been a "holder", the filter would have returned both pods and the queue scorer would have been free to pick 2pxst. So the EPP's residency mirror must have listed **6 adapters as resident on a 3-slot pod** for the whole dispatch window (~100 s). This is the specification's declared degradation **D12** ("the mirror stores anyway, so a pod's list can *briefly* run over its slot count … the mirror OVERSTATES holding"; `algorithms/routetoholder.go:215-220`; plugin `store()`, `routetoholder.go:307-326`). In this cell it was not brief: deferred requests stay in flight, so they keep their adapters pinned, and the over-capacity list is never trimmed. This is deduced from routing decisions. The mirror itself is logged only at V(4)/V(5) (`routetoholder.go:170,221`) and the EPP ran at `--v=3`, so the mirror contents were not observed directly. **What to do:** to confirm, rerun the treatment with EPP `--v=5` to log the mirror's `resident` list. Then decide whether D12's "briefly" holds at this load, since it changes how the treatment's outcome (fewer loads / lower tail) should be read. Without a simulation of this cell it is unknown whether BLIS (whose Store refuses at capacity) would have split the adapters 3 / 6.

---
## 5c. EPP Runtime Proof — Treatment Active

**What:** from EPP logs, the configured profile is the one that ran, the treatment filter actually ran and narrowed candidates, the EPP saw both pods, and the hardware matched across arms.
**Expected:** baseline = stock profile (queue-scorer 2, kv-cache-utilization-scorer 2, prefix-cache-scorer 3, max-score-picker); routetoholder = the same with `route-to-holder-filter` (fallbackSlots 3, baseModels [`ibm-granite/granite-3.3-2b-instruct`]) first (`routetoholder/routetoholder_config.yaml`).

**Effective config (log lines `runner.go` "EPP config after phase two" / "parsed config"):**

| | baseline (`...-5b9c84f9ff-b9r66`) | routetoholder (`...-666d8b6ff6-xkdt7`) |
|---|---|---|
| image / build | `ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0`, commit a5cbe600, build-ref v0.11.0 | `ghcr.io/kalantar/llm-d-router:43f299e0b7dd-routetoholder`, commit-sha "unknown", build-ref "" |
| Filters | `[utilization-detector]` | `[route-to-holder-filter, utilization-detector]` |
| Scorers | queue 2, kv-cache-utilization 2, prefix-cache 3 | queue 2, kv-cache-utilization 2, prefix-cache 3 |
| Picker / handler | max-score-picker / single-profile-handler | max-score-picker / single-profile-handler |
| filter params (Effective raw configuration) | — | `{"baseModels":["ibm-granite/granite-3.3-2b-instruct"],"fallbackSlots":3}` |
| Flow control | "Flow Control layer is disabled via the flowControl feature gate" | same |
| `Flags processed` difference | `allow-experimental-plugins: false` | `allow-experimental-plugins: true` — **the only flag that differs** |
| verbosity | `--v=3` (`"v":3`) | `--v=3` (`"v":3`) |

`utilization-detector` is a system default the loader appends in both arms.

**Runtime behaviour:**

| check | baseline | routetoholder | verdict |
|---|---|---|---|
| "Running filter plugin" lines | 1170 (1 per request: utilization-detector) | 2368 = 2 × 1184 (route-to-holder + utilization-detector) | PASS |
| `remainingEndpoints` after filters — adapter trace requests | 2 for 950/950 | **1 for 950/950** | PASS (filter narrows to a holder) |
| `remainingEndpoints` — base-model requests | 2 for 161/161 | 2 for 175/175 | PASS (baseModels bypass works) |
| plugin's own log lines | n/a | none: they are V(4)/V(5) (`routetoholder.go:170,221`) and the EPP ran at `--v=3` | expected at this verbosity |
| EPP restarts / panics / ERROR lines | 0 / 0 / 0 | 0 / 0 / 0 | PASS |
| pods discovered ("Pod added"; distinct `endpoint` in "Request handled") | 2 (qwncq 10.128.5.63, 9q5qd 10.128.6.188) | 2 (2pxst 10.128.5.75, x8rtb 10.131.6.112) | PASS |
| dispatch outcomes | 1170 handled, 0 shed (no flow control) | 1184 handled, 0 shed | PASS |
| `vllm:cache_config_info` num_gpu_blocks / block_size (EPP endpoint data does not log CacheNumBlocks at v=3) | 53,572 / 16 on both pods | 53,572 / 16 on both pods | PASS (parity) |

5c.1 (flow control active), 5c.2 (priority bands) and 5c.3's admission behaviour: **INAPPLICABLE** — this is a routing algorithm with no flowControl, priority bands or InferenceObjectives in either arm. Flow control is disabled in both, and every "Request handled" line has `priority: 0`.

**Cross-phase deployment:** separate deployments. EPP pod-template-hash 5b9c84f9ff vs 666d8b6ff6. The decode template hash is the same (844f997b87), but the pods are different: baseline on pokprod-b93r44s2 + b93r38s1, routetoholder on b93r44s2 + b93r43s1. Same namespace `kalantar-0`. Hardware config is identical: H100 80GB HBM3, 53,572 GPU blocks, 857,152 KV tokens, same vLLM args.

**Verdict: PASS (both cells)** — the treatment filter is loaded, sits first in the profile's Filters, runs on every request, and narrows every adapter request to one pod. The baseline runs the stock profile. Limitation 3 holds: the experimental-plugins flag is the only EPP flag that differs.

### 5c.6 InferenceObjective Resolution at Runtime

**SKIP — no objectives configured.** `baseline_config.yaml:8` says "config.md declares no InferenceObjectives, so none are added"; neither overlay has `inferenceObjectives` / `extraObjects`.
Verbosity prerequisite still checked, because the arms came from different deployments: the `"objectiveKey"` field appears in 19,890 (baseline) and 21,312 (routetoholder) EPP lines. Both EPPs are at `--v=3`, which meets the ≥ 3 requirement. Every "Request handled" line has `objectiveKey: ""`, `priority: 0`. "No associated InferenceObjective found, using default" appears 1170 / 1184 times, once per request, as expected with no objectives.

---

## 5d. Prefix Hit Ratio

**What:** prefix caching is off, so there should be no prefix hits.
**Expected:** `enable_prefix_caching: False` (config.md `--no-enable-prefix-caching`).
**Actual:** `enable_prefix_caching: False` in all 4 logs. Every `loggers.py` line says `Prefix cache hit rate: 0.0%` (23/24/21/19 lines). The final scrape has `vllm:prefix_cache_queries_total 0` and `vllm:prefix_cache_hits_total 0` on all 4 pods. The EPP's `prefix-cache-scorer` is a separate routing index and stays on in both arms at weight 3 (config.md §2).
**Verdict: PASS (both cells).**

---

## 5e. Request Status

**What:** status of every trace request, overall and per adapter.
**Expected:** ~100% `status=ok` in both arms; no shedding (no admission control).
**Actual (`trace_data.csv`):**

| arm | total | ok | error | shed | ok % | finish_reason | thermal flag |
|---|---|---|---|---|---|---|---|
| baseline | 950 | 950 | 0 | 0 | 100% | length 950 | none (5h SKIP) |
| routetoholder | 950 | 950 | 0 | 0 | 100% | length 950 | none (5h SKIP) |

Per adapter, all 9 adapters are 100% ok in both arms (column "ok n / ok %" in 5b.1-C). Server side: 0 non-200 access lines. `vllm:request_success_total` has no `abort`/`error` finishes.
**Verdict: PASS (both cells).**

---

## 5f. Error Analysis

**What:** counts of ERROR / WARNING / OOM / CUDA error / timeout / Traceback in each vLLM log, plus EPP error-level lines.

| log | ERROR | WARNING | OOM | CUDA error | timeout | Traceback |
|---|---|---|---|---|---|---|
| baseline 9q5qd | 0 | 21 | 0 | 0 | 0 | 0 |
| baseline qwncq | 0 | 21 | 0 | 0 | 0 | 0 |
| routetoholder 2pxst | 0 | 21 | 0 | 0 | 0 | 0 |
| routetoholder x8rtb | 0 | 21 | 0 | 0 | 0 | 0 |
| EPP (both) | 0 (only DEBUG/INFO lines) | — | — | — | — | — |

The 21 warnings are identical in every log and harmless: 10 "Unknown vLLM environment variable" (llm-d chart env names), enforce-eager/inductor notices, LoRA dev-endpoint notices, the LoRA tokenizer deprecation, and 3 `jit_monitor` "Triton kernel JIT compilation during inference" (`_lora_expand_kernel`, `_lora_shrink_kernel`, `_gumbel_sample_kernel`). The LoRA kernel JITs fire on the first LoRA request, during preflight (baseline 19:46:24–26, routetoholder 20:15:05–08). The `_gumbel_sample_kernel` JIT fires during warmup (19:47:32; 20:16:12–13). Both happen before the first trace send (19:47:36.9; 20:16:17.4).
**Verdict: PASS (both cells).**

---

## 5g. KV Cache Utilization

| arm | pod | GPU KV cache size | Max concurrency (8,192 tok/req) | Available KV memory | num_gpu_blocks | KV usage max in window |
|---|---|---|---|---|---|---|
| baseline | 9q5qd | 857,152 tokens | 104.63x | 65.4 GiB | 53,572 | 2.2% |
| baseline | qwncq | 857,152 tokens | 104.63x | 65.4 GiB | 53,572 | 1.7% |
| routetoholder | 2pxst | 857,152 tokens | 104.63x | 65.4 GiB | 53,572 | 1.4% |
| routetoholder | x8rtb | 857,152 tokens | 104.63x | 65.4 GiB | 53,572 | 2.3% |

Identical capacity on all pods. KV cache is not a constraint in this cell (≤ 2.3% used, 0 preemptions).
**Verdict: PASS (both cells).**

---
## Descriptive context: per-adapter TTFT (no verdict)

There is no simulated latency for this cell (limitation 1), so nothing here is a parity check. TTFT = `first_chunk_time_us − send_time_us` and E2E = `last_chunk_time_us − send_time_us`, over `status=ok` rows of `trace_data.csv`. Values in ms. The pod each adapter was routed to in routetoholder is shown in brackets.

| adapter | n | baseline TTFT p50 | baseline TTFT p99 | rth TTFT p50 | rth TTFT p99 | baseline E2E p50 / p99 | rth E2E p50 / p99 |
|---|---|---|---|---|---|---|---|
| answerability [2pxst] | 326 | 131 | 44,875 | 118 | 143 | 5,344 / 53,070 | 3,528 / 19,744 |
| citations [x8rtb] | 164 | 15,521 | 67,910 | 132 | 823 | 17,772 / 77,567 | 4,913 / 28,303 |
| uncertainty [2pxst] | 96 | 33,838 | 84,540 | 116 | 140 | 38,390 / 88,776 | 3,920 / 21,963 |
| context_relevance [x8rtb] | 92 | 34,817 | 100,986 | 25,231 | 67,519 | 39,472 / 108,610 | 28,836 / 73,311 |
| hallucination_detection [2pxst] | 61 | 27,897 | 89,531 | 122 | 140 | 35,466 / 95,964 | 3,307 / 22,618 |
| query_rewrite [x8rtb] | 64 | 45,786 | 87,905 | 20,485 | 39,126 | 52,832 / 97,434 | 23,230 / 47,533 |
| answer_relevance_classifier [x8rtb] | 58 | 15,246 | 67,037 | 23,592 | 79,146 | 18,987 / 75,590 | 27,219 / 88,407 |
| requirement_check [x8rtb] | 55 | 21,875 | 66,004 | 47,164 | 94,028 | 29,192 / 76,605 | 56,018 / 107,177 |
| answer_relevance_rewriter [x8rtb] | 34 | 32,040 | 94,971 | 130 | 27,433 | 36,756 / 98,347 | 6,247 / 33,232 |
| **all** | 950 | 9,339 | 89,959 | 128 | 77,740 | 17,394 / 97,662 | 5,586 / 84,665 |

Per pod (trace rows joined to EPP decisions): baseline 9q5qd TTFT p50 17,269 / p99 84,492; qwncq 153 / 94,772. Routetoholder 2pxst 118 / 142; x8rtb 147 / 91,158.
Both arms' `saturation.json` end in composite `OVERLOADED`. The tens-of-seconds TTFTs are LoRA-slot deferral queues (5b.1-D), not GPU or KV limits. In routetoholder the tail sits entirely on x8rtb, the pod given 6 adapters for 3 slots.

---

## Section 5 verdict rollup

| subsection | baseline / granite-9adapter / i1 | routetoholder / granite-9adapter / i1 |
|---|---|---|
| 5h GPU thermal/power | SKIP (0 util==100 samples; max 36 °C, cap 700 W) | SKIP (same reason; max 34 °C, cap 700 W) |
| 5a vLLM config | PASS | PASS |
| 5b distribution | PASS (45.2 / 54.8%, spread 19.4%) | PASS (affinity; 50.8 / 49.2%; skew follows residency) |
| 5b.1-A adapter coverage | PASS (9/9, all on both pods) | PASS (9/9; 3 on 2pxst, 6 on x8rtb) |
| 5b.1-C adapter proportion | PASS (χ² 11.57, p 0.172; all within 3σ) | PASS (same trace counts) |
| 5b.1-D adapter loads/evictions | loads not countable; both pods cycle 9 adapters, deferred max 142 / 139 | **WARN**: loads not countable; mirror held 6 adapters on 3-slot x8rtb for the whole window (D12 not "brief"); deferred 0 on 2pxst vs max 128 on x8rtb |
| 5c EPP runtime proof | PASS (stock profile, v=3, 2 pods) | PASS (filter in Filters, runs every request, 950/950 narrowed to 1 pod, base-model bypass) |
| 5c.1–5c.3 flow control / priority | INAPPLICABLE (routing algorithm) | INAPPLICABLE |
| 5c.6 InferenceObjectives | SKIP (none configured; verbosity v=3 OK) | SKIP (none configured; verbosity v=3 OK) |
| 5d prefix | PASS (off, 0 hits) | PASS (off, 0 hits) |
| 5e status | PASS (950/950 ok) | PASS (950/950 ok) |
| 5f errors | PASS (0 ERROR) | PASS (0 ERROR) |
| 5g KV cache | PASS (857,152 tok, 104.63x) | PASS (identical) |

Runtime counts: baseline 9 PASS / 0 WARN / 0 FAIL; routetoholder 8 PASS / 1 WARN / 0 FAIL (plus SKIP/INAPPLICABLE rows above). GPU thermal: 0 PASS / 0 WARN / 0 FAIL / 2 SKIP. No `thermal-FAIL` cells.

**Action items from §5**
1. [WARN] routetoholder: the residency mirror stayed over capacity (6 adapters on a 3-slot pod) for the whole run, so x8rtb thrashed while 2pxst idled its queue. To observe the mirror directly, rerun with EPP `--v=5`. Then reassess D12's "briefly" under deferral-heavy load before reading the arm's outcome as the algorithm's effect.
2. [INFO] GPU adapter loads/evictions cannot be counted from vLLM v0.30.0 logs or `lora_requests_info`. If load counts are the primary outcome, collection needs a per-load signal (e.g. vLLM debug logging of LoRA activation, or a load/evict counter).
3. [INFO] `trace_data.csv:x_request_id` does not match the EPP's `x-request-id` (0/950), so trace rows can only be linked to routing decisions by time order. Propagating the client id would make the per-request join exact.
