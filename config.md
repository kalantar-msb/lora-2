# Deployment configuration — route-to-holder LoRA routing

The deployment this transfer targets and every knob the arm reads.
`/sim2real-bootstrap` parses the **vLLM Pod Configuration** table and the
`blis observe` block. Everything else is for people.

## 1. Component

| Component | Ref |
| --- | --- |
| https://github.com/llm-d/llm-d-router | v0.11.0 (a5cbe600ebade00cf3e9885beaf2bfacddeabce1) |

## 2. vLLM Pod Configuration

| Parameter | Value |
| --- | --- |
| Model | ibm-granite/granite-3.3-2b-instruct |
| GPU | H100_SXM_80GB |
| max_model_len | 8192 |
| Number of vLLM pods | 2 |
| tensor_parallel_size | 1 |
| max_num_seqs | 64 |
| block_size | 16 |
| gpu_memory_utilization | 0.90 |
| dtype | bfloat16 |
| --no-enable-prefix-caching | |
| cpu limit | 8 |
| memory limit | 32Gi |
| cpu request | 4 |
| memory request | 16Gi |

Notes:

- **The model is not in bootstrap's `MODEL_METADATA`.** That is why
  `max_model_len` is stated. Bootstrap needs a lookup entry for
  `ibm-granite/granite-3.3-2b-instruct` before Task 3 produces a usable
  baseline.
- **GPU** was set by the operator in the predecessor bundle
  (`kalantar-msb/lora`), which flagged it as a placeholder. Confirm the
  cluster's accelerator.
- **Prefix caching is off in vLLM**, as the operator chose in the predecessor
  bundle. vLLM's own prefix cache is a latency effect, not a routing one: at
  v0.7.1 it keys blocks by tokens plus the adapter's integer id
  (`vllm/sequence.py:529`). The EPP's `prefix-cache-scorer` is a separate index,
  seeded with the model name, which is the adapter name for a LoRA request
  (`pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash/hashing.go:95`).
  It is fed by routing decisions, so this flag does not turn it off. It stays on
  in both arms at the same weight. With this bundle's synthetic prompts it is
  expected to be neutral (specification header, D7).
- **2 pods** × `--max-loras 3` = 6 GPU slots for 9 adapters, so eviction is
  forced. With as many slots as adapters, every pod eventually holds everything
  and the filter has nothing to choose.

## 3. LoRA serving configuration

The generator has no aliases for these, so a row in the table above would be
skipped. They live in `inputs/lora-serving.yaml`, copied verbatim from the
predecessor bundle. Requested of bootstrap: place it at
`baselines/defaults/lora-serving.yaml`, where it merges onto the generated
baseline.

| Parameter | Value | Why |
| --- | --- | --- |
| --enable-lora | (set) | required |
| --max-loras | 3 | GPU adapter slots per pod; must equal the plugin's `fallbackSlots` |
| --max-lora-rank | 32 | highest rank in the set; sizes every slot |
| --max-cpu-loras | 6 | host cache, itself LRU (`vllm/lora/models.py:680-681` at v0.7.1). At 6 < 9 it reduces disk reloads but does not remove them |
| VLLM_ALLOW_RUNTIME_LORA_UPDATING | true | runtime adapter loading; present at v0.7.1 (`vllm/envs.py:461`) |
| VLLM_PLUGINS | lora_filesystem_resolver | **CONFIRM** against the deployed image: v0.7.1 has `VLLM_PLUGINS` but no `lora_filesystem_resolver` |
| VLLM_LORA_RESOLVER_CACHE_DIR | /adapters | **CONFIRM** against the deployed image: absent at v0.7.1. Where the init container places the 9 adapters |

## 4. Engine assumptions the arm depends on

The plugin's residency mirror assumes two vLLM behaviours. They were checked
only against vLLM v0.7.1, not against the image this deployment runs.

| Assumption | Checked at | Deployed vLLM |
| --- | --- | --- |
| `vllm:lora_requests_info`'s `running_lora_adapters` lists only adapters of running requests | v0.7.1, `vllm/engine/llm_engine.py:1628-1634` | **CONFIRM** |
| GPU adapter slots are evicted least-recently-used | v0.7.1, `vllm/lora/models.py:704-710` | **CONFIRM** |

If either is false, the mirror's idea of what each pod holds drifts from the
truth (specification header, D5).

## 5. Simulation → deployment mapping

There is no simulation run of this cell. The table pairs each flag the
lora-control harness passes to BLIS (`harness/runner.py:321-329`,
`harness/runner.py:551-558`) with the vLLM parameter it stands for, at the value
a BLIS run of this cell must use to describe the same deployment.

| BLIS flag | Value for this cell | vLLM parameter | Agrees? |
| --- | --- | --- | --- |
| `--model` | ibm-granite/granite-3.3-2b-instruct | Model | yes |
| `--hardware` / `--tp` | H100 / 1 | GPU / tensor_parallel_size | yes |
| `--num-instances` | 2 | Number of vLLM pods | yes |
| `--lora-adapter-capacity` (lora-config `adapter_capacity`) | 3 | `--max-loras` | yes |
| lora-config adapter ranks (max 32) | 32 | `--max-lora-rank` | yes |
| `--max-num-running-reqs` | **64** (BLIS default is 256, `cmd/root.go:1139`) | max_num_seqs | only if passed explicitly |
| `--block-size-in-tokens` | 16 (default, `cmd/root.go:1143`) | block_size | yes |
| `--gpu-memory-utilization` | 0.9 (default, `cmd/root.go:1198`) | gpu_memory_utilization | yes |
| `--max-model-len` | **8192** (default 0 = derived from the HF config, `cmd/root.go:1154`) | max_model_len | only if passed explicitly |
| `--routing-policy` | route-to-holder | EPP `route-to-holder-filter` + stock scorers | see the specification header, D7–D9 |
| `--snapshot-refresh-interval` | 50000 µs (default, `cmd/root.go:1195`); residency forced Immediate | EPP metrics refresh interval | residency: no (D11) |
| `--routing-deterministic-tiebreak` | off (default, `cmd/root.go:1170`) | `max-score-picker` rotates ties | no (D8) |

## 6. Simulator-only knobs

No deployment equivalent:

- The LoRA cost constants in `inputs/granite-9adapter.lora-config.yaml`
  (`load_base_latency_us`, `load_bandwidth_bytes_us`,
  `footprint_bytes_per_rank`, `step_overhead_tiers`). These are **fitted for
  Llama-3.1-8B, not granite**, as the file's own header says. Absolute
  simulated latencies from them are not meaningful. Orderings between arms at
  the same constants are.
- `--seed`, `--catalog`, `--metrics-path`, `--lora-config`.
- `--creation-policy` (on-demand, the default) and `--eviction-policy` (lru, the
  default). In the deployment both are vLLM's own behaviour.

## 7. blis observe invocation

```bash
blis observe \
  --max-concurrency 10000 \
  --timeout 1800 \
  --warmup-requests 50 \
  --prewarm-duration 60s \
  --detectors composite \
  --api-format completions \
  --record-itl
```

Streaming is forced on by `--record-itl`. The workload itself declares
`streaming: false` on every client, and the measuring binary streams a request
only if the workload asks for it, unless `--record-itl` overrides that
(`cmd/observe_cmd.go:1371-1372`, `:1559` in sim2real's `inference-sim` submodule at b79b592a; lint-skip: that path also exists in the simulation tree, so it was checked by hand). Keep `--record-itl`. A streamed response header arrives at the first
token, which is the plugin's earliest confirmation that a load finished
(specification header, D2). Without it, every request is non-streaming and that
confirmation arrives only at completion.

The workload names an adapter on every client. The pipeline renders the
per-adapter dispatch flag from that, so it is not listed here. The measuring
binary is the `blis` built from sim2real's `inference-sim` submodule (branch
`observe-lora-dispatch`, `b79b592a`), not the simulation pin above.

## 8. Per-arm settings

| Arm | EPP plugins | Differs from baseline by |
| --- | --- | --- |
| `baseline` | stock default profile: queue-scorer 2, kv-cache-utilization-scorer 2, prefix-cache-scorer 3, max-score-picker | — |
| `routetoholder` | `route-to-holder-filter` first, then the same three scorers at the same weights and max-score-picker | the filter only |

`route-to-holder-filter` parameters:

| Parameter | Value | Notes |
| --- | --- | --- |
| `fallbackSlots` | 3 | must equal `--max-loras`; used until a pod reports `max_lora` |
| `baseModels` | `["ibm-granite/granite-3.3-2b-instruct"]` | requests for these skip the filter |

The EPP must run as **one replica**. The mirror only sees traffic its own replica
routed (specification header, D1).
