# Sim2Real Parity Report

- **Real bundle**: RUN=l2-1 (`workspace/runs/l2-1`)
- **Sim reference**: `kalantar-msb/lora-2`, the specification bundle: `algorithms/routetoholder.go`, `config.md`, `workloads/granite-9adapter.workload.yaml`
- **Translation**: `43f299e0b7dde9620fdff16474347832f0b00cb3eeac7a724fc916ef730f8628`
- **Treatment image**: `ghcr.io/kalantar/llm-d-router:43f299e0b7dd-routetoholder`
- **Date**: 2026-10-07 (report directory `check/20261007T205820Z/`)

**Operator-acknowledged limitations**
1. **No simulation exists for this cell.** The cell is granite-3.3-2b, 9 adapters, 2 pods, 3 slots. The lora-control simulations used a different model, so latency parity against simulation is INAPPLICABLE. Expected values here come from the workload spec, `config.md` and the specification.
2. **Two simulator trees are involved.** The measuring binary (sim2real `inference-sim` at `b79b592a`) generated the requests. The algorithm was translated from `atantawi/inference-sim` at `c9fb6038`.
3. **The arms come from different deployments.** The baseline data predates `--allow-experimental-plugins` (EPP pod `…-5b9c84f9ff-b9r66`). The treatment ran with it (EPP pod `…-666d8b6ff6-xkdt7`).

## Summary

| Category | PASS | WARN | FAIL |
|----------|------|------|------|
| Workload (§1, incl. 1h adapter share, client side) | 10 | 6 | 0 |
| Config (§2a, 2b, 2d) | 6 | 0 | 0 |
| Variant Isolation (§2c) | 2 | 1 | 0 |
| Signal (§3) | 4 | 4 | 0 |
| Policy (§4a; 4b–4d INAPPLICABLE) | 2 | 0 | 0 |
| Runtime (§5a–5g, incl. 5b.1 adapter share, server side) | 17 | 1 | 0 |
| GPU Thermal/Power (§5h) | 0 | 0 | 0 (2 SKIP) |
| **Total** | **41** | **12** | **0** |

GPU thermal was skipped in both arms because no sample reached 100% GPU utilisation. The maximum was 36 °C at a 700 W cap, so no cell is tagged `thermal-FAIL`.

## Headline results

- **Both arms received the identical workload.** Both traces match the seed-42 generator request for request: client, adapter, arrival time, input and output tokens (§1). Any difference between the arms is due to routing.
- **Adapter coverage and share (operator's extra check): PASS in both arms.**
  - All 9 adapters received traffic, both in the trace and on the server side. Three independent server-side sources agree.
  - Per-adapter counts match `rate_fraction`: χ² = 11.57 (df 8, p = 0.172), and all 9 adapters fall within the binomial ±3σ bands. The largest gap is 1.68 pp (uncertainty).
  - 100% of requests returned `ok` for every adapter (§1h, §5b.1).
- **The treatment is active and implements the specified rule.**
  - `route-to-holder-filter` is first in the profile's filters and ran on every request.
  - It narrowed all 950 adapter requests to one pod. Base-model requests skipped the filter.
  - It matches the specification line for line (§4a, §5c).
  - The only differences from baseline are the filter, the image, and the experimental-plugins flag (§2c).
- **Routing footprint** (§5b.1-B):
  - Baseline split every adapter across both pods, so each pod cycled all 9 adapters through 3 slots.
  - The treatment partitioned the adapters strictly: 3 on `2pxst` and 6 on `x8rtb`.

## Per-iteration verdicts

```
── Per-iteration verdicts ────────────────────────────────────────
  wl-granite-9adapter.workload|baseline|i1        PASS
  wl-granite-9adapter.workload|routetoholder|i1   PASS
──────────────────────────────────────────────────────────────────
Aggregates: 2 PASS  0 FAIL  0 MISSING  0 SKIP
Exit code:  0
```

WARNs are counted as PASS in this rollup. The routetoholder WARNs below still need attention before you interpret the arm's latency.

## Action Items

1. **[WARN] The treatment's residency mirror overstated what `x8rtb` held for the whole run** (§5b.1-D, §5c).
   - **Observation.** For all six of `x8rtb`'s adapters, the filter's only candidate was `x8rtb`. That held even when `x8rtb` had over 100 requests deferred waiting for a slot and `2pxst` had none.
   - **Inference.** The mirror must have listed 6 adapters as resident on a 3-slot pod for the whole ~100 s window.
   - **Effect.** `2pxst` (3 adapters) never deferred a request and had TTFT p99 142 ms. `x8rtb` (6 adapters) deferred as much as a baseline pod and had TTFT p99 ~91 s.
   - **Specification.** This is declared degradation D12, but D12 calls the overrun "brief". Here it lasted the whole run. The likely mechanism also involves D4: pins are taken at dispatch rather than at batch admission, so requests deferred inside vLLM keep their adapters pinned, and the mirror is never trimmed.
   - **Confirm it** by re-running the treatment with EPP `--v=5`, which logs the mirror's resident list. Then decide whether D4/D12 need a specification change before this arm's latency is read as the algorithm's effect.
   - **Unknown:** without a simulation of this cell, it is not known whether BLIS, which refuses to store at capacity, would also have split the adapters 3/6.
2. **[WARN] The baseline's provenance has drifted** (§2c). `cluster/baseline.yaml` now contains `--allow-experimental-plugins`, but the baseline data came from the earlier deployment without it. The flag only gates Alpha plugins at startup (`registry.go:97-108`), so the baseline's behaviour is unaffected. Re-running the baseline under the current assembly would make the run fully reproducible from its files.
3. **[WARN] What vLLM's running-adapters label means on v0.30.0 is still unconfirmed** (§3b, D5). The deployed vLLM is v0.30.0. Runtime data are consistent with the specification's assumption: the running and waiting labels were identical in every series, and never listed more than 3 adapters. One `config.md` CONFIRM item is now settled: the filesystem resolver works ("Resolved and loaded LoRA adapter … using FilesystemResolver").
4. **[WARN] The saturation filter was active in both arms** (§3c, D9). Both arms ended `OVERLOADED` in `saturation.json`. All queueing was LoRA-slot deferral, with KV-cache use at most 2.3%. The filter can't break the holder rule, but it affected routing at this load in both arms.
5. **[WARN] Residency freshness differs from the simulator** (§3d). In the simulator residency is read Immediately; in the treatment it is not (D2/D11). The delay couldn't be measured at `--v=3`.
6. **[WARN ×3, not deployment faults] Workload WARNs** (§1, identical in both arms):
   - **1d output tokens:** the mean is 7.8% below the expected value because of the seed-42 draw. The server produced exactly the requested length.
   - **1f streaming:** `--record-itl` forces streaming even though the spec says `false`.
   - **1g burstiness:** one client's CV is 1.222 against a bound of 1.2, which is within the Poisson 99% band for its 54 gaps.
7. **[INFO] Adapter loads and evictions cannot be counted** from vLLM v0.30.0 logs or `lora_requests_info`, yet they are the mechanism's primary expected outcome. Measuring them needs a per-load signal.
8. **[INFO] The trace can't be joined to routing decisions by request ID.** `trace_data.csv:x_request_id` doesn't match the EPP's `x-request-id` (0 of 950), so joins currently rely on time order.

## Section reports

- [§1 Workload parity](section1-workload-parity.md)
- [§2–4 Config, signal and policy parity](section234-config-signal-policy.md)
- [§5 Runtime analysis](section5-runtime-pd.md)
