# §2–§4 Config, Signal and Policy Parity — run l2-1

Scope: the two PRESENT rows, `baseline | granite-9adapter.workload | i1` and `routetoholder | granite-9adapter.workload | i1`. This is a ROUTING experiment. The checks written for admission control are adapted as the context file directs.

Paths used below:
- `$E` = `/Users/kalantar/projects/go.workspace/src/github.com/kalantar-msb/lora-2`
- `$RUN_DIR` = `$E/workspace/runs/l2-1`
- `$B` = `$RUN_DIR/results/baseline/granite-9adapter.workload/i1`
- `$T` = `$RUN_DIR/results/routetoholder/granite-9adapter.workload/i1`
- `$GEN` = `$E/workspace/translations/43f299e0…8628/generated`
- `$LLMD` = `$E/llm-d-router`
- `$ASIM` = `/Users/kalantar/projects/go.workspace/src/github.com/atantawi/inference-sim`, the algorithm-source simulator at c9fb603

**Operator-acknowledged limitations. These are stated, not failed.**
1. No simulation exists for this cell. "Expected" values come from `$E/config.md`, the spec `$E/algorithms/routetoholder.go`, and the workload spec, not from simulated output.
2. Two BLIS trees exist: the measuring binary and the algorithm source. §3 and §4 cite the algorithm source `$ASIM`.
3. Baseline ran on the first deployment (EPP `…-5b9c84f9ff-b9r66`, without `--allow-experimental-plugins`). Treatment ran on the second (EPP `…-666d8b6ff6-xkdt7`, with the flag). This was verified from logs; see 2c.

**Tree note.** The context says `$LLMD` has the generated plugin applied. On disk it does not: `$LLMD` is clean at `a5cbe600` (`git describe` = `v0.11.0`), and `pkg/epp/framework/plugins/scheduling/filter/routetoholder/` is absent there. Stock-code proofs below cite `$LLMD`. Plugin proofs cite `$GEN/routetoholder/…`.

## Verdict rollup

| Check | baseline/i1 | routetoholder/i1 |
|---|---|---|
| 2a vLLM server config | PASS | PASS |
| 2b Instance count | PASS | PASS |
| 2c Variant isolation | PASS (one provenance WARN note) | PASS (one provenance WARN note) |
| 2d EPP config vs config.md §8 | PASS | PASS |
| 3a Signal availability | PASS | PASS |
| 3b Signal transform / semantics | PASS (KV/queue units; baseline reads no ActiveModels) | WARN (ActiveModels semantics on vLLM v0.30.0 unconfirmed, D5) |
| 3c Saturation stage | WARN (declared D9 is active at runtime) | WARN (same) |
| 3d Snapshot delay | PASS (50ms metrics refresh = sim 50ms; no residency reader) | WARN (residency not Immediate; declared D2/D11) |
| 4a Routing policy | PASS (stock profile) | PASS (Filter matches the spec rule) |
| 4b–4d Admission / priority tiers | INAPPLICABLE | INAPPLICABLE |

There are no FAIL items.

---

## 2. CONFIG PARITY

### 2a. vLLM server config

**What was checked.** The vLLM engine settings actually used, for both arms and all four pods, against `config.md` §2 (vLLM Pod Configuration) and §3 (LoRA serving).

**Sources.**
- `trace_header.yaml` of each arm gives `server.model: ibm-granite/granite-3.3-2b-instruct`. It carries no other engine fields.
- Server logs, line 7 `non-default args:` and line 26 `Initializing a V1 LLM engine (v0.30.0) with config:`.
- `resources/pod-*decode*.yaml` gives the rendered `vllm serve` argv, env, limits, and node affinity.
- `gpu_logs/*.log` line 1.

The four server logs are `$B/server_logs/…-9q5qd_vllm.log`, `…-qwncq_vllm.log`, `$T/server_logs/…-2pxst_vllm.log` and `…-x8rtb_vllm.log`. Their `non-default args` lines are byte-identical apart from the timestamp:

```
non-default args: {'model_tag': '/model-cache/models/ibm-granite/granite-3.3-2b-instruct', 'host': '0.0.0.0', 'port': 8200,
 'model': '/model-cache/models/ibm-granite/granite-3.3-2b-instruct', 'dtype': 'bfloat16', 'max_model_len': 8192,
 'enforce_eager': True, 'served_model_name': ['ibm-granite/granite-3.3-2b-instruct'], 'block_size': 16,
 'gpu_memory_utilization': 0.9, 'enable_prefix_caching': False, 'enable_lora': True, 'max_loras': 3,
 'max_lora_rank': 32, 'max_cpu_loras': 6, 'max_num_seqs': 64}
```

| Param | Expected (config.md) | Actual: baseline (9q5qd, qwncq) | Actual: routetoholder (2pxst, x8rtb) | Match |
|---|---|---|---|---|
| Model | ibm-granite/granite-3.3-2b-instruct (§2) | served_model_name same; weights at /model-cache/models/ibm-granite/granite-3.3-2b-instruct | same | PASS |
| tensor_parallel_size | 1 (§2) | 1 (engine config line 26) | 1 | PASS |
| GPU | H100_SXM_80GB (§2) | `GPU 0: NVIDIA H100 80GB HBM3` (gpu_logs); nodeAffinity `NVIDIA-H100-80GB-HBM3` | same | PASS (HBM3 is the SXM part) |
| max_model_len | 8192 (§2) | 8192 | 8192 | PASS |
| gpu_memory_utilization | 0.90 (§2) | 0.9 | 0.9 | PASS |
| enable_prefix_caching | off, `--no-enable-prefix-caching` (§2) | False | False | PASS |
| max_num_seqs | 64 (§2) | 64. The flag `--max-num-seqs=64` overrides env `VLLM_MAX_NUM_SEQ=256`, which vLLM logs as an unknown env var | 64 | PASS |
| max_num_batched_tokens | not stated | 8192 (log line 20: "Chunked prefill is enabled with max_num_batched_tokens=8192") | 8192 | PASS (not specified; identical across arms) |
| enable_chunked_prefill | not stated | True (engine config) | True | PASS (not specified; identical) |
| block_size | 16 (§2) | 16 | 16 | PASS |
| dtype | bfloat16 (§2) | bfloat16 | bfloat16 | PASS |
| cpu / memory limit, request | 8 / 32Gi, 4 / 16Gi (§2) | limits cpu 8, memory 32Gi; requests cpu 4, memory 16Gi (pod yaml l.174-182) | same | PASS |
| --enable-lora | set (§3) | enable_lora True | True | PASS |
| --max-loras | 3 (§3) | 3; vLLM metric label `max_lora="3"` in every `vllm:lora_requests_info` series (318 series) | 3 (122 series) | PASS |
| --max-lora-rank | 32 (§3) | 32 | 32 | PASS |
| --max-cpu-loras | 6 (§3) | 6 | 6 | PASS |
| VLLM_ALLOW_RUNTIME_LORA_UPDATING | true (§3) | `true` (pod env l.138-139) | `true` | PASS |
| VLLM_PLUGINS | lora_filesystem_resolver (§3, marked CONFIRM) | set (l.140-141), and it works at runtime: `Resolved and loaded LoRA adapter 'answer_relevance_rewriter' using FilesystemResolver` (9q5qd l.1877) | same (2pxst l.2124, x8rtb l.2027) | PASS. The CONFIRM is resolved on v0.30.0 |
| VLLM_LORA_RESOLVER_CACHE_DIR | /adapters (§3, CONFIRM) | `/adapters` (l.142-143); `pull-lora` init container exitCode 0 on all 4 pods | same | PASS |
| vLLM image | not pinned (config.md §4 cites v0.7.1 for semantics only) | docker.io/vllm/vllm-openai:v0.30.0 | same | Note: the deployed engine is v0.30.0. See 3b for what this means for D5 |

The rendered argv is also identical in all four pods (pod yaml `vllm serve … --enable-lora --max-loras=3 --max-lora-rank=32 --max-cpu-loras=6 --max-num-seqs=64 --no-enable-prefix-caching --dtype=bfloat16`). It adds `--enforce-eager`, which config.md does not mention. Both arms carry it, so it does not break the A/B comparison.

**Verdict: PASS for both rows.**

Side observation: baseline pod `9q5qd` started vLLM at 19:43:13, 12 minutes after `qwncq` at 19:31:07, because its `pull-lora` clone ran from 19:30:31 to 19:43:13. Its restartCount is 0, and both pods were up before the trace began (`trace_header.created_at` 19:50:12; first EPP request 19:46:16).

### 2b. Instance count

| Arm | Expected (config.md §2 "Number of vLLM pods") | `server_logs/` files | decode Deployment `replicas` | Match |
|---|---|---|---|---|
| baseline | 2 | 2 (9q5qd, qwncq) | 2 (`cluster/baseline.yaml`, `replicas: 2`) | PASS |
| routetoholder | 2 | 2 (2pxst, x8rtb) | 2 (`cluster/routetoholder.yaml:40`) | PASS |

EPP replicas: 1 in both arms (`resources/deployment-*-router-epp.yaml:21`, `replicas: 1`). This meets config.md §8 ("The EPP must run as one replica"; spec D1).

The decode Deployment spec is identical between arms. Diffing `resources/deployment-…-decode.yaml` with only volatile metadata and status stripped gives no differences.

Pods landed on different nodes. Baseline used `pokprod-b93r38s1` and `pokprod-b93r44s2`; treatment used `pokprod-b93r43s1` and `pokprod-b93r44s2`. The GPU model is the same on all of them.

**Verdict: PASS for both rows.**

### 2c. Variant isolation (adapted for a routing filter)

**What was checked.** That baseline and routetoholder differ only in three ways: the `route-to-holder-filter` plugin entry, the EPP image that contains it, and the `--allow-experimental-plugins` flag (limitation 3).

**Expected difference (by design).** One plugin entry plus its `pluginRef` placed first in the `default` profile, and the treatment image. For a routing filter the admission-oriented rows of the spec table (`flowControl`, `router.inferenceObjectives`) are expected to be absent in both arms.

**(i) Generated overlays.** `diff` of the inner `custom-plugins.yaml` text, `$GEN/baselines/baseline/baseline_config.yaml:15-38` against `$GEN/routetoholder/routetoholder_config.yaml:23-52`:

```
4a5,9
>           - type: route-to-holder-filter
>             parameters:
>               fallbackSlots: 3
>               baseModels:
>               - "${model.name}"
17a23
>             - pluginRef: route-to-holder-filter
```

**(ii) Assembled cluster files.** `diff $RUN_DIR/cluster/baseline.yaml $RUN_DIR/cluster/routetoholder.yaml` shows two differences:
- `router.epp.image` is added in treatment (`registry: ghcr.io/kalantar`, `repository: llm-d-router`, `tag: 43f299e0b7dd-routetoholder`, `pullPolicy: Always`).
- The `pluginsCustomConfig` string differs exactly as in (i).

Everything else is identical: decode block, LoRA flags, env, init container, model, proxy, tokenizer and vllmCommon. The two `pipelinerun-*.yaml` files differ only in run name, phase value, and the same embedded scenario difference.

**(iii) Effective EPP config at runtime (EPP logs).**

| Item | baseline (`$B/epp_logs/…-b9r66_*.log`) | routetoholder (`$T/epp_logs/…-xkdt7_*.log`) |
|---|---|---|
| `GIE build` | `runner/runner.go:222`, commit-sha `a5cbe600…`, build-ref `v0.11.0` | `runner/runner.go:223`, commit-sha `unknown`, build-ref `""` |
| Image (pod yaml) | `ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0` (sha256:39257d24…) | `ghcr.io/kalantar/llm-d-router:43f299e0b7dd-routetoholder` (sha256:c0f0e092…) |
| `Loaded raw configuration` (`loader/configloader.go:99`) | `{Plugins: [queue-scorer, kv-cache-utilization-scorer, prefix-cache-scorer, max-score-picker, metrics-data-source{…}, core-metrics-extractor], SchedulingProfiles: [default: queue-scorer 2.00, kv-cache-utilization-scorer 2.00, prefix-cache-scorer 3.00, max-score-picker]}` | `{Plugins: [{Type: route-to-holder-filter, Parameters: {"baseModels":["ibm-granite/granite-3.3-2b-instruct"],"fallbackSlots":3}} + the same six], SchedulingProfiles: [default: route-to-holder-filter, queue-scorer 2.00, kv-cache-utilization-scorer 2.00, prefix-cache-scorer 3.00, max-score-picker]}` |
| Effective config after system defaults (`configloader.go:168`) | the plugins above + single-profile-handler, fcfs-ordering-policy, global-strict-fairness-policy, static-usage-limit-policy, 4 parsers, utilization-detector; profile ends `… max-score-picker, utilization-detector` | identical injected set; profile `route-to-holder-filter, …, max-score-picker, utilization-detector` |
| `EPP config after phase two` (`runner.go:391` / `:392`) | `Filters: [utilization-detector]`, Scorers `queue 2, kv 2, prefix 3`, Picker `max-score-picker`, `FlowControlConfig:<nil>` | `Filters: [route-to-holder-filter, utilization-detector]`, same scorers, picker, `FlowControlConfig:<nil>` |
| featureGates | `{}` (Flags processed) | `{}` |
| InferenceObjectives | none. Every request logs "No associated InferenceObjective found, using default" (1170×) | none (1184×) |

**`${model.name}` substitution.** Confirmed. The treatment overlay has `baseModels: ["${model.name}"]` (`routetoholder_config.yaml:31`), and the EPP loaded `"baseModels":["ibm-granite/granite-3.3-2b-instruct"]`, which equals vLLM's `served_model_name`. Runtime consequence: 175 base-model requests in the treatment arm, during prewarm from 20:14:48 to 20:16:10, matched `baseModels` and so did not enter the residency mirror (generated plugin `routetoholder.go:149`, `:184`).

**EPP command-line flags.** The full JSON `flags` objects from the `Flags processed` line (`runner.go:246` / `:247`) were diffed key by key. Exactly one key differs:

```
< allow-experimental-plugins false     (baseline, b9r66)
> allow-experimental-plugins true      (routetoholder, xkdt7)
```

All other flags are equal, including `refresh-metrics-interval 50ms`, `metrics-staleness-threshold 2s`, `refresh-prometheus-metrics-interval 5s`, `v 3`, `feature-gates {}`, `ha-enable-leader-election false` and `config-file /config/custom-plugins.yaml`.

The pod args agree. Baseline `…-b9r66` has no `--allow-experimental-plugins`; treatment `…-xkdt7` has `--allow-experimental-plugins=true`. The EPP Deployment diff shows only that arg and the image.

**Effect of `--allow-experimental-plugins` (code proof).** It is a startup gate only.
- `$LLMD/pkg/epp/framework/interface/plugin/registry.go:97-108`:

  ```go
  func ValidatePluginStability(handle Handle, allowExperimentalPlugins bool) error {
      ...
      if stability == StabilityAlpha && !allowExperimentalPlugins {
          return fmt.Errorf("plugin '%s' (type: '%s') has Alpha stability level, but command line flag --allow-experimental-plugins is not set", ...)
  ```

- It is called only from `cmd/epp/runner/runner.go:465` and `:1096`.
- The generated registration is `fwkplugin.Register(routetoholder.RouteToHolderType, fwkplugin.StabilityAlpha, routetoholder.Factory)` (`$GEN/routetoholder/cmd/epp/runner/runner.go:578`). That is why the treatment needs the flag.
- Baseline's effective plugin set (above) contains no route-to-holder plugin, so for the baseline the flag changes nothing at runtime. Limitation 3 therefore has no behavioural effect.

**Treatment image provenance.** The treatment image reports `commit-sha unknown`. Two pieces of indirect evidence show it is v0.11.0 plus the generated `runner.go`:
- `diff $LLMD/cmd/epp/runner/runner.go $GEN/routetoholder/cmd/epp/runner/runner.go` is exactly two added lines: the import at 124 and `Register` at 578.
- Every runner.go `caller` line the treatment EPP logged is shifted by exactly those insertions. Stock 222/246/363/391 become 223/247/364/392 (+1, after the import). Stock 760/827 become 762/829 (+2, after both). The generated file holds the same statements at those lines.

The treatment EPP also ran `route-to-holder-filter` as a filter 1184 times, once per scheduled request (`scheduling/scheduler_profile.go:169` "Running filter plugin", `plugin: route-to-holder-filter/route-to-holder-filter`). This proves the image contains the plugin.

**Section-by-section match table.**

| Section | Must be identical | Identical? |
|---|---|---|
| featureGates | yes | yes (`{}` both) |
| plugins (types) | all except the filter | yes. Only `route-to-holder-filter` added |
| schedulingProfiles: scorers, weights, picker | yes | yes (queue 2, kv 2, prefix 3, max-score-picker) |
| injected defaults (utilization-detector, profile handler, parsers) | yes | yes |
| flowControl | n/a for a routing filter | `FlowControlConfig:<nil>` in both. INAPPLICABLE as the independent variable |
| router.inferenceObjectives | yes | none in both |
| EPP flags | yes | yes, except `allow-experimental-plugins` (limitation 3, proven inert for baseline) |
| EPP image | expected to differ | differs as designed |
| Instance count | yes | 2 / 2 |
| vLLM config | yes | yes (2a) |

**Provenance WARN, not an isolation failure.** The on-disk `$RUN_DIR/cluster/baseline.yaml:79` has `allow-experimental-plugins: true`. It comes from `$E/baselines/defaults/epp-verbosity.yaml:25`, added after the baseline run. The baseline that actually produced `$B` ran without it. So the assembled baseline file no longer reproduces the deployment that produced the baseline data. Re-assembling and re-deploying would give a baseline EPP with the flag set. By the code above that should be behaviourally identical, but it is not the deployment that was measured.

**Verdict: PASS for both rows.** The only differences are the filter entry, the image, and the inert `allow-experimental-plugins` flag. The provenance note above is a WARN.

### 2d. EPP config vs `config.md` §8 (correctness)

`config.md` has no `## llm-d EPP Configuration — Baseline|Treatment` blocks and no `kind: InferenceObjective`. Under the spec's literal condition 2d would be SKIP. Adapted: the source of truth is the `## 8. Per-arm settings` table (`config.md` §8).

| Field path | config.md §8 value | Generated (overlay) | Effective (EPP log) | Match |
|---|---|---|---|---|
| apiVersion / kind | (component.kind EndpointPickerConfig, README "Required shape") | `llm-d.ai/v1alpha1` / `EndpointPickerConfig` (both overlays) | same ("Raw config after phase one") | PASS |
| baseline profile | queue-scorer 2, kv-cache-utilization-scorer 2, prefix-cache-scorer 3, max-score-picker | `baseline_config.yaml:30-38` the same | same | PASS |
| treatment profile | `route-to-holder-filter` first, then the same three scorers at the same weights, then max-score-picker | `routetoholder_config.yaml:43-52` the same | same | PASS |
| treatment differs by | "the filter only" | diff in 2c(i) is the filter only | same | PASS |
| `fallbackSlots` | 3; must equal `--max-loras` | `3` (`routetoholder_config.yaml:29`) | `"fallbackSlots":3`; vLLM `max_lora="3"` | PASS |
| `baseModels` | `["ibm-granite/granite-3.3-2b-instruct"]` | `["${model.name}"]` | `["ibm-granite/granite-3.3-2b-instruct"]` | PASS (after substitution) |
| EPP replicas | one | (chart) | Deployment `replicas: 1` both arms | PASS |
| featureGates | none stated | none | `{}` | PASS |
| InferenceObjectives | none declared (baseline overlay header, l.8) | none | none | PASS |

Both overlays also list `metrics-data-source` and `core-metrics-extractor`, which §8 does not mention. They are identical in both arms, and the loader would inject them anyway (`baseline_config.yaml:5-7` comment).

**Verdict: PASS for both rows.**

---

## 3. SIGNAL PARITY

The algorithm reads these signals:
- the request's adapter
- per-pod residency, held in a mirror that is corrected by ActiveModels
- slots per pod (MaxActiveModels, with FallbackSlots until it appears)

The scorers it relies on read WaitingQueueSize, KVCacheUsagePercent and the prefix index. The spec is `$E/algorithms/routetoholder.go`. The generated plugin is `$GEN/routetoholder/pkg/epp/framework/plugins/scheduling/filter/routetoholder/routetoholder.go` (called `gen` below).

### 3a. Signal availability (code proof)

| Signal | Sim (`$ASIM`) | LLMD source | Generated plugin |
|---|---|---|---|
| Request adapter | `req.Adapter` (`sim/routing.go:267-270`) | `pkg/epp/framework/interface/scheduling/types.go:50-51` `// TargetModel is the final target model after traffic split.` `TargetModel string`. Set at `pkg/epp/requestcontrol/director.go:229` `TargetModel: reqCtx.TargetModelName`, which defaults to the body's model at `:335-336` | `gen:149-152` `request.TargetModel` |
| Residency ("holds a") | `snap.ResidentAdapters[adapter]` (`sim/routing.go:279`) | no direct signal (spec ELIMINATIONS, README Observability "estimated") | mirror `podMirror.lru` (`gen:126-132`), tested by `isResident` (`gen:329-336`) |
| ActiveModels (correction) | — | `pkg/epp/framework/interface/datalayer/metrics.go:27-28` `// ActiveModels holds only adapters that have at least one running or queued request.` `ActiveModels map[string]int`; filled at `pkg/epp/framework/plugins/datalayer/extractor/metrics/extractor.go:249-258` from label `running_lora_adapters` (`:48`) | `gen:299-304` |
| MaxActiveModels (slots) | lora-config `adapter_capacity` | `metrics.go:33-34` `// MaxActiveModels is the maximum number of adapters the model server can load (max_lora).`; filled from label `max_lora` (`extractor.go:50`, `:260-267`) | `gen:296-298`, `slotsOf` `gen:371-376` |
| LoRA metric family | — | `factories.go:97` `LoRASpec: "vllm:lora_requests_info"`. Latest series picked by value, which is its creation timestamp (`loraspec.go:58-78`). Absent until the first adapter loads (`loraspec.go:53-57`) | FallbackSlots until then (`gen:57-60`) |
| WaitingQueueSize | queue-depth scorer | `metrics.go:36`; from `vllm:num_requests_waiting` (`factories.go:94`, `extractor.go:130-137`) | not read by the plugin; read by queue-scorer and utilization-detector |
| KVCacheUsagePercent | kv-utilization scorer | `metrics.go:37`; from `vllm:kv_cache_usage_perc` (`factories.go:96`, `extractor.go:148-155`) | not read by the plugin; read by kv-cache-utilization-scorer and utilization-detector |
| Metrics freshness stamp | — | `metrics.go:44-45` `UpdateTime time.Time`; set when any field updates (`extractor.go:210-211`) | `gen:292-295`: the correction is applied once per `UpdateTime` |
| Priority | n/a | n/a. No InferenceObjectives; routing does not read priority | n/a |

**Can the plugin actually read Metrics?** Filters receive scoped endpoints (`pkg/epp/scheduling/scheduler_profile.go:176`). The scope wrapper restricts only attribute `Get`/`Put`. `GetMetrics()` passes through: `pkg/epp/datalayer/endpoint_scope.go:120` `func (e *ScopedEndpoint) GetMetrics() *fwkdl.Metrics { return e.inner.GetMetrics() }`. At runtime the treatment EPP logged 0 "Rejected access" lines and 0 ERROR lines.

Runtime presence of the signals in vLLM's scrapes (`metrics/raw/*decode*_metrics.log`):
- baseline: 318 `vllm:lora_requests_info` series, all `max_lora="3"`
- treatment: 122 series, all `max_lora="3"`
- `vllm:kv_cache_usage_perc` and `vllm:num_requests_waiting` are present on every scrape

**Verdict: PASS for both rows.**

### 3b. Signal transform / units / semantics

**KVCacheUsagePercent (range 0–1).**

| Source | Code / value |
|---|---|
| LLMD populate | `extractor.go:152` `clone.KVCacheUsagePercent = extractValue(metric)`, raw gauge, no scaling |
| vLLM emitted (this run) | `vllm:kv_cache_usage_perc{…} 0.005880047040376368` (`$B/metrics/raw/…-9q5qd_1791402535_metrics.log`). Range over all scrapes: baseline 0.0–0.0187, treatment 0.0–0.0128. This is a fraction |
| LLMD scorer | `scorer/kvcacheutilization/kvcache_utilization.go:88` `scores[endpoint] = 1 - endpoint.GetMetrics().KVCacheUsagePercent` |
| LLMD detector | `saturationdetector/utilization/detector.go:224` `podMetrics.KVCacheUsagePercent < kvLimit`, with `kvLimit = KVCacheUtilThreshold*(1+Headroom)` (`:213`); default `DefaultKVCacheUtilThreshold float64 = 0.8 // (0.0 to 1.0)` (`config.go:32-33`) |
| LLMD tests | `kvcache_utilization_test.go:39-41` `KVCacheUsagePercent: 0.8` / `0.5` / `0.0` |
| Sim | kv-utilization `1 - used/total` (spec l.58, citing `sim/routing_scorers.go:269-275`) |
| Generated plugin | does not read KV. No transform to get wrong |

These are consistent: the value is a fraction everywhere and nothing divides by 100. **PASS.**

**WaitingQueueSize.**
- `extractor.go:134` `clone.WaitingQueueSize = int(extractValue(metric))`
- The queue scorer min-max normalises over the candidate set, `queuedepth/queue.go:85-115` `float64(maxQueueSize-…WaitingQueueSize) / float64(maxQueueSize-minQueueSize)`, returning 1.0 when all are equal.
- The sim also min-maxes over the restricted set (spec l.55-57).
- The plugin does not read it. **PASS.**

**ActiveModels / MaxActiveModels transform (the plugin's own reads).**
- `gen:296-298`: `if met.MaxActiveModels > 0 { m.slots = met.MaxActiveModels }`. This matches spec `algorithms/routetoholder.go:399-401`.
- `gen:299-304` stores every non-base adapter in `ActiveModels`, touching it if already resident. The spec does the same at `:402-411`: touch if resident, else `storeAt`. The generated `store` (`gen:312-326`) does touch-if-resident internally, so the two are equivalent.
- Nothing is removed on a scrape (`gen:287-290`; spec `:390-393`).
- Units: adapter names are strings, and the slot count is an integer from the `max_lora` label. **PASS.**

**ActiveModels semantics. WARN (CONFIRM still open, D5).**

The spec's correction step assumes `running_lora_adapters` lists only adapters of running requests. The spec checked this at vLLM v0.7.1 (`config.md` §4, marked CONFIRM). The deployed engine is v0.30.0. Evidence:
- LLMD's own field comment says the opposite: "ActiveModels holds only adapters that have at least one **running or queued** request" (`metrics.go:27`).
- In this run, every `vllm:lora_requests_info` series has `running_lora_adapters` == `waiting_lora_adapters`: 318/318 baseline and 122/122 treatment.
- No series ever lists more than 3 adapters, the `max_lora` value. Running-list size distribution: baseline 1:55, 2:113, 3:150; treatment 1:47, 2:23, 3:52. This is consistent with the running list holding only slot-resident adapters, even while baseline pods had 117–139 requests waiting with reason `deferred`.

No vLLM v0.30.0 source is available locally, so this cannot be proven from code. If the label does include adapters that are only queued, the mirror would mark a not-yet-loaded adapter resident. That OVERSTATES holding: a request is sent to a "holder" and cold-loads there. The runtime data (never more than 3) suggest the assumption holds, but it remains unconfirmed.

**Verdict for 3b: baseline/i1 PASS (KV and queue units consistent; baseline does not read ActiveModels). routetoholder/i1 WARN (ActiveModels semantics on the deployed vLLM v0.30.0 unconfirmed; runtime data consistent with the assumption).**

### 3c. Saturation stage (utilization-detector filter)

The sim has no saturation stage. The real EPP appends `utilization-detector` as the last filter in every profile, in both arms. This was declared as D9.

**Injection (code).** `$LLMD/pkg/epp/config/loader/defaults.go:276-309`, `ensureSaturationDetector`. It defaults `SaturationDetector.PluginRef` to `utilization-detector`, and, because that plugin is a `fwksched.Filter`, calls `injectFilterIntoProfiles` (`:311-324`), which appends it after the configured plugins:

```go
if !found {
    profiles[i].Plugins = append(profiles[i].Plugins, configapi.SchedulingPlugin{PluginRef: pluginRef})
}
```

The runtime order confirms it. Treatment `Filters: [route-to-holder-filter, utilization-detector]`; baseline `Filters: [utilization-detector]` ("EPP config after phase two"). Filter run counts: baseline utilization-detector 1170; treatment route-to-holder-filter 1184 and utilization-detector 1184.

**Filter function (code)**, `pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/detector.go:207-232`:

```go
qLimit := float64(d.config.QueueDepthThreshold) * (1.0 + d.config.Headroom)
kvLimit := d.config.KVCacheUtilThreshold * (1.0 + d.config.Headroom)
filtered := make([]fwksched.Endpoint, 0, len(endpoints))
for _, endpoint := range endpoints {
    podMetrics := endpoint.GetMetrics()
    if podMetrics == nil || time.Since(podMetrics.UpdateTime) > d.config.MetricsStalenessThreshold {
        continue
    }
    if float64(podMetrics.WaitingQueueSize) < qLimit && podMetrics.KVCacheUsagePercent < kvLimit {
        filtered = append(filtered, endpoint)
    }
}
if len(filtered) == 0 {
    return endpoints
}
return filtered
```

The pool-level `Saturation()` (`detector.go:133-183`) is `avg(max(queue/5, kv/0.8))` and returns 1.0 for an empty pool (`:134-140`). In this run it is unused for admission, because `FlowControlConfig:<nil>` in both arms. Only the Filter path is active.

**Thresholds.**

| Param | LLMD default (`utilization/config.go`) | Effective (EPP log `detector.go:96`, both arms) |
|---|---|---|
| queueDepthThreshold | `DefaultQueueDepthThreshold int = 5` (l.31) | 5 |
| kvCacheUtilThreshold | `0.8` (l.33) | 0.8 |
| metricsStalenessThreshold | `200 * time.Millisecond` (l.36) | 200ms |
| stalenessPolicy | — | saturated |
| headroom | — | 0 |

**Interaction with the holder rule.** The detector filters the set route-to-holder returned and falls back to that same input when it would empty it (`:228-230`). So it can only narrow within the holders, never add a non-holder. The holder rule survives. This matches the spec's D9 claim, which cites `detector.go:207-232`; that citation is correct at a5cbe600.

**Runtime: the stage is active in this run.** From `vllm:num_requests_waiting` in `metrics/raw` (15 s scrapes):

| Arm | Pod | scrapes with waiting > 5 | max waiting |
|---|---|---|---|
| baseline | 9q5qd | 9/15 | 135 |
| baseline | qwncq | 9/15 | 139 |
| routetoholder | 2pxst | 0/14 | 0 |
| routetoholder | x8rtb | 7/14 | 125 |

The baseline waiting requests are reported as `num_requests_waiting_by_reason{reason="deferred"}`, for example 117 at `$B/…-9q5qd_1791402535`, not `capacity`. KV stays below 0.02 throughout.

So the queue threshold (5), not KV, is what fires. During those intervals the detector removes a saturated pod whenever another candidate is unsaturated, and returns all candidates when both are. Two consequences:
- In baseline it can override the scorers.
- In treatment it can drop a saturated holder only when another holder exists.

The sim has no such stage.

**Verdict: WARN for both rows.** The code matches the spec's declaration (D9) and cannot break the holder rule. However, this declared, sim-absent stage was actively filtering for large parts of both runs. Bias and runtime causes belong to §5.

### 3d. Snapshot delay / signal freshness

| Signal | Sim (`$ASIM`) | Real (EPP) |
|---|---|---|
| Residency | Immediate. `sim/cluster/cluster.go:444-448`: `if config.RoutingPolicy == "route-to-holder" { obsConfig.PinResidentAdaptersImmediate() }` | Mirror updated on request events: PreRequest at dispatch (`gen:183-224`), first 2xx ResponseHeader (`gen:229-241`), and ResponseBody end of stream (`gen:248-265`). ActiveModels correction sampled only inside Filter, at most once per `UpdateTime` (`gen:164`, `gen:291-295`) |
| Queue, KV, ActiveModels and other metrics | `--snapshot-refresh-interval` default `50000` µs (`cmd/root.go:1195`, "default 50ms = llm-d parity"); config.md §5 sets 50000 µs | `refresh-metrics-interval` = 50ms (EPP "Flags processed", both arms). Default and floor: `pkg/epp/server/options.go:48` `MinRefreshMetricsInterval = 50 * time.Millisecond`, `:154` `RefreshMetricsInterval: MinRefreshMetricsInterval`; values below are clamped (`:415-418`) |
| Staleness cut-off | none | `metrics-staleness-threshold` 2s (flag, `options.go:156`). The utilization-detector uses its own 200ms (3c) |
| Scrape cost / jitter | instant | HTTP scrape of `/metrics` through `metrics-data-source` (same params both arms) |

For non-residency signals, real and sim match at 50ms. **PASS.**

For residency, the sim reads ground truth with zero delay. The real mirror learns of a completed load only at the first successful response or a later ActiveModels sample. That is always after the true completion, as declared in D2 and D11. During that window a second request for the same adapter sees no holder, falls back to all pods, and may load a second copy. The treatment workload streams (`--record-itl`, config.md §7), so confirmation (a) arrives at first token, not at completion. That keeps the window as short as the design allows.

**Unobservable at this log level.** The plugin logs its holder count at `V(logutil.DEBUG)` (`gen:170-171`) and its mirror state at `V(logutil.TRACE)` (`gen:221-222`). Those are levels 4 and 5 (`pkg/common/observability/logging/const.go:20-23`). The EPP ran at `--v=3`, so neither decision-level line is in the logs. The size of the D2/D11 window cannot be measured from this run's EPP logs.

**Verdict: baseline/i1 PASS** (50ms metrics refresh equals the sim's 50ms; baseline reads no residency). **routetoholder/i1 WARN** (residency is not Immediate; declared D2/D11; the window is not measurable at v=3).

---

## 4. POLICY PARITY

### 4a. Routing policy (code proof)

**Sim policy.** `--routing-policy route-to-holder` (config.md §5) wrapping the default weighted scorer. `$ASIM/sim/routing.go:263-301`:

```go
if adapter != "" {
    holders := make([]RoutingSnapshot, 0, len(state.Snapshots))
    for _, snap := range state.Snapshots {
        if snap.ResidentAdapters[adapter] { holders = append(holders, snap) }
    }
    if len(holders) > 0 {
        restricted := &RouterState{Snapshots: holders, ...}
        return r.inner.Route(req, restricted)
    }
}
// No holder (or base-model request) → unconstrained fallback (D1).
return r.inner.Route(req, state)
```

The inner scorer weights are `precise-prefix-cache 2, queue-depth 1, kv-utilization 1` (`sim/routing_scorers.go:100-106`). It is an argmax with random tie-break: `bestIdx = tied[ws.rng.Intn(len(tied))]` (`sim/routing.go:207-230`).

**Real pipeline, both arms.** Filters run, then scorers, then the picker (`$LLMD/pkg/epp/scheduling/scheduler_profile.go:128-143`):

```go
endpoints := p.runFilterPlugins(ctx, request, candidateEndpoints)
if len(endpoints) == 0 { return nil, errcommon.Error{Code: errcommon.ServiceUnavailable, ...} }
weightedScorePerEndpoint := p.runScorerPlugins(ctx, request, endpoints)
result := p.runPickerPlugin(ctx, request, weightedScorePerEndpoint)
```

Filters run in order, each on the previous one's output (`:166-189`, `filteredEndpoints = datalayer.Unscope(filter.Filter(ctx, request, scoped))`). Scorers see only the filtered set and sum `enforceScoreRange(score) * scorer.Weight()` (`:236-252`).

| Plugin | What it does (LLMD code) |
|---|---|
| queue-scorer (w=2) | min-max of `WaitingQueueSize` over candidates; 1.0 if all equal (`scorer/queuedepth/queue.go:85-115`) |
| kv-cache-utilization-scorer (w=2) | `1 - KVCacheUsagePercent` (`scorer/kvcacheutilization/kvcache_utilization.go:85-91`) |
| prefix-cache-scorer (w=3) | EPP approximate prefix index (`scorer/prefix/plugin.go:63`). The EPP logged `blockSizeTokens 16` raised to the 64 minimum (`approximateprefix/plugin.go:167`) in both arms |
| max-score-picker | stable sort by score descending, then round-robin rotation within equal-score tiers (`picker/maxscore/picker.go:95-118`): `// RotateScoredEndpoints provides deterministic round-robin tie-breaking for equal-score candidates.` |
| utilization-detector (injected) | filter, see 3c |
| route-to-holder-filter (treatment only) | below |

**Generated Filter vs the spec decision rule.** Spec `algorithms/routetoholder.go:284-305` against `gen:148-177`:

| Rule element | Spec | Generated | Match |
|---|---|---|---|
| Base-model request is unconstrained | `:286-288` `if p.isBaseModel(adapter) { return endpoints }` | `gen:149-151` `if request == nil \|\| p.isBaseModel(request.TargetModel) { return endpoints }` | yes |
| Adapter read from TargetModel | `:285` | `gen:152` | yes |
| Holders collected in input order | `:293-300` loop over `endpoints`, `applyScrape`, then `isResident` | `gen:157-168`, same loop, same order. Endpoints with nil metadata are skipped (defensive; `md.ID` is needed to key the mirror) | yes |
| Holders exist: return only holders | `:301-303` | `gen:173-175` | yes |
| No holder: return every endpoint | `:304` | `gen:176` | yes |
| Adds no score of its own | wrapper adds none (spec l.22-23) | Filter only; it does not implement Scorer (compile-time assertions `gen:74-77`: Filter, PreRequest, ResponseHeaderProcessor, ResponseBodyProcessor) | yes |
| Loading adapter is not a holder | `isResident` checks `lru` only (`:446-453`) | `gen:329-336`, identical | yes |

**Generated mirror vs the spec (the bridge that decides who is a holder).**

| Element | Spec | Generated | Match |
|---|---|---|---|
| Cold-miss eviction at dispatch: if `len(lru)+len(loading) >= slots`, evict LRU unpinned; mark loading; pin | `:325-342` | `gen:204-219` | yes |
| Re-used request id is aborted first | — | `gen:198-202` (extra safety; releases the old pin and clears its pending load) | addition, consistent with D13 |
| Confirmation (a): first 2xx response header | `:347-356` | `gen:229-241`; `servedOK` = 2xx from `:status` or `status` header (`gen:426-442`) | yes; resolves spec unknown `servedOK` (`:547-556`) |
| End of stream: store if a natural, successful end and unconfirmed; else clear pending unless another request is pending; release pin | `:363-388` | `gen:248-265` using `response.TerminationCause == fwkrc.TerminationCauseNatural && servedOK(response)`, then `clearPendingLoad` (`gen:270-274`) and `release` (`gen:277-285`) | yes |
| Confirmation (b): ActiveModels once per UpdateTime; never removes | `:394-412` | `gen:291-305` | yes |
| Store: evict LRU unpinned to capacity; overrun if all pinned (D12) | `:426-440` | `gen:312-326` | yes |
| LRU helpers (touch / lruUnpinned / evict / slotsOf / mirrorFor / otherPending) | `:457-519` | `gen:339-399` | identical logic |
| `primaryTarget` = `ProfileResults[PrimaryProfileName].TargetEndpoints[0]` | `:536-541` (unknown) | `gen:406-419` | yes; resolves the spec unknown |

**Hook wiring (code).** Every configured plugin is added to request control under each interface it implements:
- `cmd/epp/runner/runner.go:799` `r.requestControlConfig.AddPlugins(handle.GetAllPlugins()...)`
- `pkg/epp/requestcontrol/request_control_config.go:119-127`: PreRequest goes to `preRequestPlugins`, ResponseHeaderProcessor to `responseReceivedPlugins`, ResponseBodyProcessor to `responseStreamingPlugins`.

Response headers reach plugins unfiltered: `pkg/epp/handlers/response.go:135-139` copies every Envoy response header into `reqCtx.Response.Headers` before `director.HandleResponseHeader`. The `:status`/`status` check in `gen:430` therefore has the header to read.

**Equivalence: sim route-to-holder vs real route-to-holder-filter + scorers + max-score-picker.**
- Decision rule (hard restriction to holders, else all; base model unconstrained): **equivalent.**
- Declared non-equivalences, all in the spec:
  - Residency is estimated, not read (D1–D6, D11–D13).
  - Inner weights: real 2/2/3 is 0.29/0.29/0.43; sim 1/1/2 is 0.25/0.25/0.50. The real prefix index is approximate (D7).
  - Tie-break: real is round-robin, sim is random (D8).
  - The utilization-detector stage exists only in the real system (D9).
- Baseline is stock weighted scoring with no LoRA awareness, which is the intended control (config.md §8).

**Runtime confirmation.**
- Treatment: `route-to-holder-filter/route-to-holder-filter` ran 1184 times, once per scheduled request, as the first filter.
- 175 base-model requests in the treatment were passed through unfiltered by design, because `baseModels` substitution worked (2c).
- Per adapter, the treatment EPP handled exactly the same counts as the baseline EPP: 341 answerability, 174 citations, 107 uncertainty, 96 context_relevance, 67 query_rewrite, 66 hallucination_detection, 61 answer_relevance_classifier, 59 requirement_check, 38 answer_relevance_rewriter. That is 1009 adapter requests per arm, from `targetModelName` on the "Request handled" lines. So the filter received the same adapter stream in both arms.
- Per-decision holder counts are not logged at `--v=3` (3d), so the fraction of requests that had a holder cannot be shown from EPP logs. §5's per-adapter × per-pod matrix from server logs is the indirect evidence.

**Verdict: PASS for both rows.** Baseline: the stock profile is as configured. Treatment: the generated Filter matches the spec decision rule line for line, and the declared differences are D7–D9.

### 4b. Admission policy: baseline. INAPPLICABLE

This is a routing experiment with no admission plugin in either arm. Both EPPs report `FlowControlConfig:<nil>` ("EPP config after phase two"), and `featureGates {}`. The injected `static-usage-limit-policy`, `fcfs-ordering-policy` and `global-strict-fairness-policy` appear in both arms' effective plugin lists, but flow control is disabled, so they are not in the request path.

### 4c. Admission policy: treatment. INAPPLICABLE

`route-to-holder-filter` is a scheduling Filter plus request-control observers. It never rejects or sheds a request. Its parameters are `fallbackSlots` and `baseModels`, checked in 2d. There are no admission thresholds or ramps.

### 4d. Priority tier mapping. INAPPLICABLE

There are no priority bands. No InferenceObjectives are declared (config.md; baseline overlay l.8) or deployed: every request logs "No associated InferenceObjective found, using default". The generated plugin has no priority switch.
