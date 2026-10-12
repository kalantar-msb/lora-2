# 1. Workload Parity

**Run:** l2-1 | **Rows checked:** `baseline / granite-9adapter.workload / i1`, `routetoholder / granite-9adapter.workload / i1`
**Expected-value source:** `$WORKLOADS_DIR/granite-9adapter.workload.yaml` (`EXPERIMENT_ROOT/workloads/`). `$SIM` is the same experiment root, so there is no second copy to diff. The YAML-parsed file is **identical** to the `workloadSpec` param in both PipelineRuns (`cluster/pipelinerun-granite-9adapter.workload|{baseline,routetoholder}|i1.yaml`), which is what was mounted as `/workspace/workload.yaml`.
**Measuring binary:** BLIS `inference-sim` @ `b79b592a` (matches `blisGitCommit` in both PipelineRuns). All source citations below are to that tree.

**Acknowledged limitations (not failures):**
1. No simulation run exists for this cell (granite-3.3-2b, 9 adapters, 2 pods, 3 slots). "Expected" values come from the workload spec and the BLIS sampler code, not from a simulated trace.
2. The measuring/workload-generating BLIS tree (`b79b592a`) differs from the algorithm-source tree (`atantawi/inference-sim c9fb6038`). Only the measuring tree matters for §1, since it generated the requests.
3. Baseline and routetoholder ran on different EPP deployments (see §2/§5); this does not affect the client-side workload, which §1 shows to be identical in both arms.

**Method.** Scripts: `/tmp/s2r-check-scripts/a1_analyze.py`, `a1_tables.py` (run with `python3 -I`). In addition to the spec's checks, the exact request sequence was **regenerated** with the measuring BLIS generator (`/tmp/s2r-check-scripts/regen/main.go` calls `workload.GenerateWorkload(spec, MaxInt64, num_requests)` — the same eager path `blis observe` uses at `cmd/observe_cmd.go:572`) and compared request-for-request against each trace.

## 1.0 Cross-cutting evidence: the trace is exactly the generated workload

| Check | baseline | routetoholder |
|---|---|---|
| rows in `trace_data.csv` | 950 | 950 |
| `request_id` coverage | 50..999, no gaps, no duplicates | 50..999, no gaps, no duplicates |
| rows whose `client_id` / `adapter` / `arrival_time_us` / `input_tokens` / `output_tokens` differ from the regenerated seed-42 request with the same index | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| identical request content to the other arm (same 5 fields, keyed by `request_id`) | yes | yes |
| `output_tokens` (server-reported completion tokens) == requested `max_tokens` | 950 / 950 | 950 / 950 |
| `status=ok` | 950 / 950 | 950 / 950 |

In plain terms: both arms sent the same 950 requests, with the same timing, sizes, and adapters, and that sequence is exactly what the BLIS generator produces for this spec and seed. Any BLIS run of this spec would see the same requests. `output_tokens` equals the requested length because observe sets `min_tokens = max_tokens` (`cmd/observe_cmd.go:1546-1552`); every request finished with `finish_reason=length`.

Warmup: the first 50 dispatched requests (generator index 0-49, `--warmup-requests 50`) are excluded from the trace by index (`cmd/observe_cmd.go:1228-1230`). The generator sorts by arrival time before indexing (`sim/workload/generator.go:307-314`), so the excluded requests are the first 6.16 s of arrivals.

## Phase: baseline / Workload: granite-9adapter.workload / i1  and  Phase: routetoholder / Workload: granite-9adapter.workload / i1

Both rows carry identical request content (§1.0), so each table below shows both rows side by side. Verdicts are given per row; where the two rows differ, the table says so.

### 1a. Arrival Rate (QPS) — baseline: PASS | routetoholder: PASS

**What:** whether requests were scheduled at the spec rate, and whether the load generator actually sent them on schedule.
**Expected:** `aggregate_rate: 10.0` req/s (`$WORKLOADS_DIR/granite-9adapter.workload.yaml`).
**Actual:** intended QPS = 950 / (max − min `arrival_time_us`); actual QPS = 950 / (max − min `send_time_us`), from each `trace_data.csv`.

| Metric | Spec (`$WORKLOADS_DIR`) | baseline | routetoholder |
|---|---|---|---|
| intended QPS (arrival_time_us) | 10.0 | 9.650 (98.446 s span) | 9.650 (98.446 s span) |
| diff vs spec | — | −3.5% | −3.5% |
| actual QPS (send_time_us) | — | 9.650 (98.446 s span) | 9.650 (98.446 s span) |
| actual vs intended | — | +0.00% | +0.00% |
| send lag vs schedule (send − arrival, minus its minimum), p50 / p99 / max | — | 0.58 / 1.09 / 1.32 ms | 0.57 / 1.09 / 2.35 ms |
| peak client-side requests in flight | cap `--max-concurrency 10000` | 334 | 181 |

**Verdict: PASS (both).** The scheduled rate is 3.5% under 10 req/s, inside the 5% tolerance; it is the random realization of Poisson arrivals for seed 42 (the full 1000-request sequence, warmup included, realizes 9.56 req/s over 104.6 s). Every request left the client within 2.4 ms of its scheduled time, so the client was never the bottleneck — the concurrency cap was far above the peak in flight.
*Context for §5 (does not gate §1):* the peak in-flight counts (334 baseline, 181 routetoholder) show requests piling up at the server, and both `saturation.json` reports end `composite: OVERLOADED`. The server saw the intended load but could not keep up with it; that is Agent 3's subject.

### 1b. SLO Class Distribution → mapped to per-client `rate_fraction` — baseline: PASS | routetoholder: PASS

**Mapping:** this workload has no `slo_class` (the column is empty in all 950 rows of both traces, and no client sets one). Per the run instructions, 1b is applied to the thing `rate_fraction` actually controls: the share of requests per `client_id`. Each client names exactly one adapter and `client_id == "client-" + adapter` holds for every traced row, so this is the same measurement as the adapter-proportion half of 1h; the full table is in 1h and is not repeated.

**Expected:** each client's `rate_fraction` (Zipf α=1.0 over 9; sums to 1.0000). The generator turns these into independent per-client Poisson rates (`sim/workload/client.go:40-50`, `aggregate_rate * fraction / total`), merges and sorts them, then truncates to `num_requests` (`sim/workload/generator.go:307-314`). The per-client counts are therefore a multinomial sample.
**Actual:** `client_id` counts in `trace_data.csv`.

| Gate | baseline | routetoholder |
|---|---|---|
| spec tolerance: every client within 2 pp | yes — largest gap 1.68 pp (client-uncertainty) | yes — same |
| chi-square goodness of fit, 8 dof | χ² = 11.57, p = 0.172 | χ² = 11.57, p = 0.172 |
| every client within its binomial ±3σ band | yes — largest \|z\| = 2.06 (client-requirement_check) | yes — same |

**Verdict: PASS (both).** The client mix matches the spec under both the fixed 2 pp tolerance and the statistical tests. Note that 2 pp is only 1.3σ for the largest client at n = 950 (σ = 1.55 pp), so the fixed tolerance is stricter than the sampling noise allows; the statistical test is the more meaningful gate here.

### 1c. Input Token Distribution — baseline: PASS | routetoholder: PASS

**What:** whether prompt lengths follow the configured distribution.
**Expected:** all 9 clients: `input_distribution: gaussian {mean: 256, std_dev: 64, min: 16, max: 2048}`. BLIS `GaussianSampler.Sample` (`sim/workload/distribution.go:24-35`) draws `N(mean, sd)`, **clamps to [min, max] before rounding** (line 29), rounds (line 30), and floors at 1 (lines 31-33); `NewLengthSampler` requires all four params for gaussian (lines 229-238). Moments come from a 200,000-draw Monte Carlo that reproduces those steps exactly. The clamp hardly matters here: only 0.009% of draws fall below 16 (z = −3.75), so the clamped value and the closed form agree.
**Actual:** the `input_tokens` column (the generator's sampled length; `cmd/observe.go:674-693` records it as suffix-only, which equals the total because `prefix_length` = 0).

*Caption: expected values from clamped MC (N = 200k); closed form 256 / 64.*

| Stat | Expected (clamped MC) | baseline | diff | routetoholder | diff | Verdict |
|---|---|---|---|---|---|---|
| mean | 256.21 | 254.76 | −0.56% | 254.76 | −0.56% | PASS (tol 5%) |
| stdev | 63.96 | 66.10 | +3.34% | 66.10 | +3.34% | PASS (tol 10%) |
| min | 16 (clamp floor) | 16 (1 request at the floor) | — | 16 | — | consistent |
| max | 540 in 200k draws (bound 2048) | 468 | — | 468 | — | consistent |

**Verdict: PASS (both).** For context, the server tokenized these prompts to `server_input_tokens` mean 255.79 (about +1.0 token per request over `input_tokens`, from prompt-to-token conversion); this is informational only.

### 1d. Output Token Distribution — baseline: WARN | routetoholder: WARN

**What:** whether response lengths follow the configured distribution.
**Expected:** all 9 clients: `output_distribution: exponential {mean: 128}`, no min/max. BLIS `ExponentialSampler.Sample` (`sim/workload/distribution.go:44-57`) draws `Exp × mean`, **rounds first** (line 46), applies min/max only if set (lines 47-52; neither is set here, see `NewLengthSampler` lines 240-254), and floors at 1 (lines 53-55). Clamped MC with N = 200k gives the moments below. The floor at 1 lifts the 0.39% of draws that round to 0, so the MC mean (127.44) is slightly below the closed-form 128 (rounding and the floor partly cancel).
**Actual:** the `output_tokens` column = server-reported completion tokens (`cmd/observe.go:406`/`489`). This equals the requested `max_tokens` for 950/950 rows in both arms (§1.0).

*Caption: expected values from clamped MC (N = 200k, round → floor at 1); closed form 128 / 128.*

| Stat | Expected (clamped MC) | baseline | diff | routetoholder | diff | Verdict |
|---|---|---|---|---|---|---|
| mean | 127.44 | 117.54 | **−7.77%** | 117.54 | **−7.77%** | outside tol 5% |
| stdev | 127.46 | 111.07 | **−12.86%** | 111.07 | **−12.86%** | outside tol 10% |
| min | 1 | 1 (15 requests = 1.6%; MC: 1.1% equal 1) | — | 1 | — | consistent |
| max | 1519 in 200k draws (unbounded) | 643 | — | 643 | — | consistent |

**Verdict: WARN (both).** The 5% / 10% tolerances are exceeded, but **not because of the deployment**. The server produced exactly the requested length for every request, and those lengths are exactly what the BLIS generator draws for seed 42 (§1.0). So a BLIS run of this same spec would see the same 117.5-token mean. The gap is a low random draw for this seed: the standard error of the mean at n = 950 is 4.1 tokens, so 117.5 is 2.4 standard errors low, and the stdev is 2.8 standard errors low. (The full 1000-request sequence, warmup included, has mean 118.4.) Both arms share the identical sequence, so this does not bias the baseline-vs-routetoholder comparison.
**What it means / what to do:** the cell is running about 8% fewer decode tokens than the nominal `mean: 128`. Keep that in mind when comparing to any future simulation or to nominal-rate capacity estimates. If an exact nominal mean matters, raise `num_requests` or choose a different seed; no deployment fix is needed.

### 1e. Prefix Tokens — baseline: PASS | routetoholder: PASS

**What:** whether any shared-prefix tokens were sent.
**Expected:** no client sets `prefix_group` or any prefix config in the spec → `prefix_length` = 0 everywhere.
**Actual:** `prefix_length` column and `prefix_group` column.

| | baseline | routetoholder |
|---|---|---|
| All zero `prefix_length` | yes (950/950) | yes (950/950) |
| distinct `prefix_group` values | only empty | only empty |

**Verdict: PASS (both).**

### 1f. Streaming — baseline: WARN | routetoholder: WARN

**What:** whether requests used streaming responses as the spec says.
**Expected:** `streaming: false` for all 9 clients → 0% streaming.
**Actual:** `streaming` column: `true` for 950/950 rows in both arms; `num_chunks > 1` for 950/950 (the responses really were streamed).

| | Spec | baseline | routetoholder |
|---|---|---|---|
| streaming fraction | 0% | 100% | 100% |

**Why it differs:** both PipelineRuns pass `--record-itl` in `observeArgs`. That flag "forces streaming per request" (`cmd/observe_cmd.go:203`); the dispatcher sets `nextReq.Streaming = true` on every non-streaming request (`cmd/observe_cmd.go:1371-1373`). The generator itself kept `Streaming = false` (`sim/workload/generator.go:299`, `Streaming: client.Streaming`; the regenerated sequence shows `false` for all 1000).
**Verdict: WARN (both).** This is a deliberate harness override, applied identically to both arms, and it is what makes TTFT and inter-token latency measurable. It does not change the token work the server does. But the delivered workload is streaming while the spec says non-streaming. If a simulation of this cell is run later, either set `streaming: true` in the spec or note that the sim and the real deployment differ on this field.

### 1g. Burstiness / Arrival Pattern — baseline: WARN | routetoholder: WARN

**What:** whether each client's gaps between requests look like a Poisson process (CV of inter-arrival time ≈ 1).
**Expected:** every client has `arrival.process: poisson` → per-client CV in 0.8–1.2. BLIS draws each client's gaps independently with `PoissonSampler.SampleIAT` = `ExpFloat64() / rate` (`sim/workload/arrival.go:24-30`, selected at line 228), each from its own seeded RNG (`sim/workload/generator.go:224-231`). The streams are merged by a stable sort on arrival time (`generator.go:307-310`).
**Actual:** for each `client_id`, sort by `arrival_time_us`, take successive differences, CV = stdev / mean. CV on `send_time_us` is also shown, to confirm what the server actually received. The last column is a 99% band for the sample CV of that many truly exponential gaps (4,000-trial Monte Carlo), to separate real non-Poisson behaviour from small-sample noise.

| client_id | configured process | gaps (n-1) | baseline CV (arrival) | baseline CV (send) | routetoholder CV (arrival) | routetoholder CV (send) | MC 99% band for Poisson at this n | verdict |
|---|---|---|---|---|---|---|---|---|
| client-answerability | poisson | 325 | 0.952 | 0.952 | 0.952 | 0.952 | [0.87, 1.15] | PASS |
| client-citations | poisson | 163 | 1.030 | 1.030 | 1.030 | 1.030 | [0.82, 1.25] | PASS |
| client-uncertainty | poisson | 95 | 1.031 | 1.031 | 1.031 | 1.031 | [0.78, 1.30] | PASS |
| client-context_relevance | poisson | 91 | 1.110 | 1.110 | 1.110 | 1.110 | [0.77, 1.29] | PASS |
| client-hallucination_detection | poisson | 60 | 0.828 | 0.828 | 0.828 | 0.828 | [0.72, 1.36] | PASS |
| client-query_rewrite | poisson | 63 | 0.823 | 0.823 | 0.823 | 0.823 | [0.73, 1.37] | PASS |
| client-answer_relevance_classifier | poisson | 57 | 1.017 | 1.017 | 1.017 | 1.018 | [0.71, 1.37] | PASS |
| client-requirement_check | poisson | 54 | 1.222 | 1.222 | 1.222 | 1.222 | [0.72, 1.40] | WARN |
| client-answer_relevance_rewriter | poisson | 33 | 1.008 | 1.008 | 1.008 | 1.008 | [0.65, 1.43] | PASS |

Merged-stream CV — informational; does not gate verdict: baseline 1.011 (arrival) / 1.011 (send); routetoholder 1.011 / 1.011.

**Verdict: WARN (both)** — 8 of 9 clients PASS. `client-requirement_check` has CV 1.222, just above the 1.2 bound. It has only 54 gaps, and 1.222 sits well inside the 99% band for a true Poisson stream of that size ([0.72, 1.40]). This is small-sample noise in the seed-42 draw, not a sign that arrivals are clustered. Arrival and send CVs agree to 3 decimals for every client, and §1.0 shows the arrival times equal the generator's. So the dispatcher passed the Poisson pattern through unchanged, identically in both arms. No action is needed.

### 1h. Adapter Traffic Coverage & Proportion (client side) — baseline: PASS | routetoholder: PASS

*Operator-requested extra check. This is the client-side half; the server-side half (vLLM logs / `vllm:lora_requests_info`, the per-adapter × per-pod matrix) is in §5 (Agent 3).*

**What:** whether every adapter that should get traffic did get traffic, in the expected proportion, and whether its requests succeeded.
**Expected:** each client's `rate_fraction` from `$WORKLOADS_DIR/granite-9adapter.workload.yaml`. All 9 are non-zero; Zipf α=1.0; they sum to 1.0000. Expected count = 950 × fraction. The ±3σ band uses the binomial σ = √(n·p·(1−p)).
**Actual:** the `adapter` column of `trace_data.csv`, written because the run used `--dispatch-adapters` (present in both PipelineRuns' `observeArgs`). On the wire, the request's `model` field is the adapter id: `dispatchModelFor` returns the adapter when it is set (`cmd/observe.go:140-145`, used at line 193). With the flag set, the gate leaves the adapter in place (`cmd/observe.go:153-158`). The trace's `model` column still records the base model (`ibm-granite/granite-3.3-2b-instruct`) for all rows; that column is the spec's client model, not the wire value.

**Coverage:** all 9 adapters appear in both arms. No adapter has zero requests, and no adapter appears that the spec does not list.

| adapter (client_id = client-<adapter>) | rate_fraction | expected count (n=950) | 3σ band | baseline obs | baseline share | Δ pp | z | routetoholder obs | routetoholder share | Δ pp | z | within 3σ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answerability | 0.3535 | 335.8 | [291.6, 380.0] | 326 | 34.32% | -1.03 | -0.67 | 326 | 34.32% | -1.03 | -0.67 | yes |
| citations | 0.1767 | 167.9 | [132.6, 203.2] | 164 | 17.26% | -0.41 | -0.33 | 164 | 17.26% | -0.41 | -0.33 | yes |
| uncertainty | 0.1178 | 111.9 | [82.1, 141.7] | 96 | 10.11% | -1.68 | -1.60 | 96 | 10.11% | -1.68 | -1.60 | yes |
| context_relevance | 0.0884 | 84.0 | [57.7, 110.2] | 92 | 9.68% | +0.85 | +0.92 | 92 | 9.68% | +0.85 | +0.92 | yes |
| hallucination_detection | 0.0707 | 67.2 | [43.5, 90.9] | 61 | 6.42% | -0.65 | -0.78 | 61 | 6.42% | -0.65 | -0.78 | yes |
| query_rewrite | 0.0589 | 56.0 | [34.2, 77.7] | 64 | 6.74% | +0.85 | +1.11 | 64 | 6.74% | +0.85 | +1.11 | yes |
| answer_relevance_classifier | 0.0505 | 48.0 | [27.7, 68.2] | 58 | 6.11% | +1.06 | +1.49 | 58 | 6.11% | +1.06 | +1.49 | yes |
| requirement_check | 0.0442 | 42.0 | [23.0, 61.0] | 55 | 5.79% | +1.37 | +2.06 | 55 | 5.79% | +1.37 | +2.06 | yes |
| answer_relevance_rewriter | 0.0393 | 37.3 | [19.4, 55.3] | 34 | 3.58% | -0.35 | -0.55 | 34 | 3.58% | -0.35 | -0.55 | yes |

| Goodness of fit (9 adapters, 8 dof) | baseline | routetoholder |
|---|---|---|
| χ² | 11.57 | 11.57 |
| p-value | 0.172 | 0.172 |
| adapters outside ±3σ | 0 of 9 | 0 of 9 |
| largest \|Δ\| | 1.68 pp (uncertainty) | 1.68 pp (uncertainty) |
| largest \|z\| | 2.06 (requirement_check) | 2.06 (requirement_check) |

**Per-adapter success (`status=ok`):**

| adapter | baseline requests | baseline status=ok | ok rate | routetoholder requests | routetoholder status=ok | ok rate |
|---|---|---|---|---|---|---|
| answerability | 326 | 326 | 100.0% | 326 | 326 | 100.0% |
| citations | 164 | 164 | 100.0% | 164 | 164 | 100.0% |
| uncertainty | 96 | 96 | 100.0% | 96 | 96 | 100.0% |
| context_relevance | 92 | 92 | 100.0% | 92 | 92 | 100.0% |
| hallucination_detection | 61 | 61 | 100.0% | 61 | 61 | 100.0% |
| query_rewrite | 64 | 64 | 100.0% | 64 | 64 | 100.0% |
| answer_relevance_classifier | 58 | 58 | 100.0% | 58 | 58 | 100.0% |
| requirement_check | 55 | 55 | 100.0% | 55 | 55 | 100.0% |
| answer_relevance_rewriter | 34 | 34 | 100.0% | 34 | 34 | 100.0% |

**Warmup exclusion:** the first 50 dispatched requests are not in the trace. Their adapter mix, taken from the regenerated sequence, is shown below, so the traced counts can be reconciled with the full 1000-request workload. The full sequence on its own also fits `rate_fraction`: χ² = 9.83, 8 dof, p = 0.277. Warmup requests were still dispatched to the server (only their trace rows are skipped, `cmd/observe_cmd.go:1228-1230`), so server-side per-adapter counts (§5) may exceed these by the warmup column. Two other kinds of traffic also reach the server: `--prewarm-duration 60s` traffic, which names the base `--model`, not an adapter (`cmd/observe_cmd.go:178`, `:685-687`), and the `--dispatch-adapters` preflight's 1-token probes for adapters not listed in `/v1/models` (`cmd/observe.go:545-576`).

| adapter | full 1000-request sequence | warmup idx 0-49 (excluded) | traced idx 50-999 |
|---|---|---|---|
| answerability | 340 | 14 | 326 |
| citations | 173 | 9 | 164 |
| uncertainty | 106 | 10 | 96 |
| context_relevance | 95 | 3 | 92 |
| hallucination_detection | 65 | 4 | 61 |
| query_rewrite | 66 | 2 | 64 |
| answer_relevance_classifier | 60 | 2 | 58 |
| requirement_check | 58 | 3 | 55 |
| answer_relevance_rewriter | 37 | 3 | 34 |
| **total** | 1000 | 50 | 950 |

**Verdict: PASS (both).** All 9 adapters received traffic. Every adapter is within its 3σ band, the overall mix is statistically consistent with the configured Zipf fractions (p = 0.17), and 100% of each adapter's requests returned `status=ok`. The per-adapter counts are identical in the two arms because the request sequence is identical (§1.0), so any server-side difference in adapter handling between baseline and routetoholder comes from routing, not from the traffic offered.

## §1 verdict rollup

| Check | baseline / granite-9adapter.workload / i1 | routetoholder / granite-9adapter.workload / i1 |
|---|---|---|
| 1a Arrival rate | PASS | PASS |
| 1b Client mix vs `rate_fraction` (no `slo_class`) | PASS | PASS |
| 1c Input tokens | PASS | PASS |
| 1d Output tokens | WARN (seed-42 draw is 7.8% / 12.9% below nominal mean / stdev; real equals generator exactly) | WARN (same) |
| 1e Prefix tokens | PASS | PASS |
| 1f Streaming | WARN (`--record-itl` forces streaming; spec says `false`) | WARN (same) |
| 1g Per-client burstiness | WARN (client-requirement_check CV 1.222 > 1.2, within Poisson 99% band) | WARN (same) |
| 1h Adapter coverage & proportion (client side) | PASS | PASS |
| **Totals** | **5 PASS / 3 WARN / 0 FAIL** | **5 PASS / 3 WARN / 0 FAIL** |
| Latency/TTFT vs simulation | INAPPLICABLE: no simulation exists for this cell (limitation 1) | INAPPLICABLE: same |

The workload offered to the two arms is request-for-request identical and matches the BLIS generator exactly, so §1 finds no workload confound in the baseline-vs-routetoholder comparison.
