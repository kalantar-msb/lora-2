# Route-to-holder LoRA routing for llm-d

A sim2real bundle for one routing rule: **send each LoRA request to a pod that
already holds its adapter in a GPU slot; if none does, route as usual.** The rule
is BLIS's `route-to-holder` policy from `atantawi/inference-sim`. It is ported to
llm-d-router as one EPP plugin that keeps its own estimate of what each pod holds.

## Mechanism

A vLLM pod has a fixed number of GPU adapter slots (`--max-loras`). A request
whose adapter is not in a slot must wait for a load, and the load evicts the
least-recently-used idle adapter. With fewer slots than adapters, any router that
ignores residency spreads every adapter across every pod, and each pod's LRU
churns.

The quantity that differs between arms is **how often a request lands on a pod
that does not hold its adapter**. Route-to-holder sends a request to a non-holder
only when no pod holds the adapter. Each adapter therefore tends to stay on the
pods it was first loaded on, and loads happen on first use or after an eviction,
not on every change of pod. The simulator states the rule in
`sim/routing.go:263-301`.

This port has no direct residency signal (below), so it estimates residency from
what it routed. **The mechanism under test is therefore the rule plus that
estimate.**

## Required shape

The decision is a **hard restriction of the candidate set**, followed by the
ordinary weighted scoring over whatever remains. It is not a score. A score can
only bias the pick, while the simulator guarantees a holder whenever one exists.

On llm-d-router v0.11.0 that is a scheduling **Filter**. The filter is placed
ahead of the stock scorers. The same object also implements the PreRequest and
response hooks that keep the residency estimate current.

- One registration, plugin type `route-to-holder-filter`.
- `component.kind` is `EndpointPickerConfig`.
- No core modification is needed.

This placement is a candidate. The method sets were read but nothing has been
compiled. The specification header carries the placement, the interfaces, and
the eliminations: why a Scorer, `label-selector-filter`, `WaitingModels`,
`ActiveModels` alone, and `/v1/models` cannot carry the decision.

### Observability

| Quantity the rule reads | Status | Source |
| --- | --- | --- |
| the request's adapter | direct | `InferenceRequest.TargetModel` |
| which adapters each pod holds | **estimated** | an EPP-side copy of the simulator's per-pod LRU, driven by the requests this plugin routed, confirmed by first response or by `ActiveModels`; declared D1–D6, D11 |
| slots per pod | direct | `MaxActiveModels` (vLLM `max_lora`); `fallbackSlots` until the metric appears |
| queue depth, KV-cache use | direct | stock `queue-scorer`, `kv-cache-utilization-scorer` |
| prefix-cache hits | degraded | EPP approximate index vs the simulator's exact one; expected neutral here (D7) |

A true residency signal, vLLM's `vllm:lora_adapter_loaded` gauge, is not merged
(`docs/llm-d-router-proposal.md:351-366` in lora-control). Once it lands, the
estimate should be replaced by it.

## Honest status

**No measurement backs the expected effect of this rule.**

- At the pinned simulator commit, every published arm uses route-to-holder. The
  arms vary only the placement policy (`docs/experiment-results.md:78-80`). So
  route-to-holder was never compared with another routing rule there.
- The only routing variation at that pin is a single-seed probe of the placement
  arms (`docs/experiment-results.md:236-238`).
- No simulation uses this bundle's cell (granite-3.3-2b, 9 adapters, 2 pods,
  3 slots).
- The LoRA cost constants available for a granite run are fitted for
  Llama-3.1-8B (`inputs/granite-9adapter.lora-config.yaml`, header).

Related evidence, none of it a measurement of this rule:

- **Soft affinity on exact residency.** At a different pin (957bc3e4),
  affinity routing cut loads from 166 to 17 against round-robin (16 adapters,
  4 pods, 8 slots). Its benefit collapsed at the knee where slots equal adapters
  per pod: 66 loads, 3.72 s worst-adapter p99 at 4 slots
  (`docs/experiment-results.md:549-552`, `:572-578`). This bundle's cell is
  below that knee: 4.5 adapters per pod against 3 slots.
- **On-demand loading against placement.** The pinned results' on-demand arm is
  route-to-holder with no placement, the same policy as this arm. At 2 slots and
  10 s epochs its mean p99 TTFT was 230.3 ms, against 294.3 for static placement
  and 353.2 for the oracle (`docs/experiment-results.md:144-145`). It lost to the
  placed arms in epoch 0 (`docs/experiment-results.md:165-169`). After that, the
  ordering at 2 slots is unresolved at n = 5 (`docs/experiment-results.md:154-163`).
- **The in-tree soft scorer on real GPUs.** `lora-affinity-scorer`, which sees
  only in-flight adapters, was indistinguishable from stock at 8 slots, 128
  adapters and 12 req/s (`docs/llm-d-router-proposal.md:499-504`).

## Pre-registered expectation

This is a hypothesis with no margin attached. It is stated against `baseline`
(stock profile, no LoRA awareness), at this bundle's workload: 10 req/s, 1000
requests, Zipf α = 1.0 over 9 adapters.

1. **Fewer adapter loads than baseline.** Baseline spreads every adapter over
   both pods; route-to-holder keeps each adapter on a pod that holds it.
2. **Lower per-adapter TTFT tail than baseline**, most for the mid- and
   low-popularity adapters. They are the ones baseline loads most often.
3. **Where it may not win:** the hottest adapter (`answerability`, 35% of
   traffic) stays on whichever pods hold it. At higher load that concentrates
   queueing. The saturation filter mitigates this only once a holder is
   overloaded (specification header, D9). At this bundle's load this is not
   expected to bind.
4. **Against `lora-affinity-scorer`** (registered as a separate byo arm by the
   operator, if wanted): fewer loads for the less popular adapters. The in-tree
   scorer sees only adapters with requests in flight, while the estimate also
   remembers idle adapters still in a slot. That is the prediction in
   `docs/llm-d-router-proposal.md:1142-1147`, applied to this rule.

## Isolation

- **`baseline` → `routetoholder`** differs by the filter only. Scorers, weights
  and picker are the same (`config.md` §8). The difference is attributable to the
  restriction plus its residency estimate, together.
- **The estimate alone is not isolated.** An ablation that restricts to
  `ActiveModels` holders only, with no remembered state, would separate the two.
  No simulation backs either variant, and this bundle does not specify that
  ablation.
- **`lora-affinity-scorer`** differs in two ways at once (soft score, and an
  in-flight-only signal), so it is context rather than an isolating comparator.

## Open questions owned by this bundle

- **How loads are counted on the cluster.** The arms are expected to differ in
  adapter loads, and vLLM v0.7.1 exposes no load counter. Measurement may have
  to infer loads from TTFT.
- **The two vLLM behaviours the estimate assumes** (`config.md` §4) are
  confirmed only at v0.7.1.
- **An offline pre-placement arm** (lora-place's solver seeding each pod) was
  deferred. It needs the traffic distribution in advance, which the EPP cannot
  know.

## Pins

| What | Repository | Ref |
| --- | --- | --- |
| simulation (algorithm source) | `atantawi/inference-sim` | `c9fb60385bb1bb4ca7b45e5915d2dfe52227b359` (tag `lora-seams-2026-08-22`) |
| target component | `llm-d/llm-d-router` | `a5cbe600ebade00cf3e9885beaf2bfacddeabce1` (v0.11.0) |
| results provenance | `tantawi/lora-control` | `08962e691542e23055cb74e4445ca085b4932368` |
| engine semantics (not the deployed version) | `vllm-project/vllm` | `33e0602e` (v0.7.1-18) |
| measuring binary | sim2real `inference-sim` submodule | `b79b592a` (branch `observe-lora-dispatch`) |

## Layout

```
algorithms/routetoholder.go     specification layer (non-compiling)
config.md                       deployment, mapping, blis observe, per-arm settings
workloads/granite-9adapter.workload.yaml
                                BLIS workload spec, copied from kalantar-msb/lora
inputs/granite-9adapter.lora-config.yaml
                                BLIS adapter registry + cost constants (Llama-fitted)
inputs/lora-serving.yaml        vLLM LoRA flags + adapter init container
sim_results/agentic-locality-reschedule/
                                lora-control results at the simulation pin
sim_results/CHECKSUMS.sha256
```

The workload file's header refers to a sibling `../lora-rag-intrinsics-zipf.yaml`
that is not part of this bundle. It was copied verbatim, so the reference is kept
as written.
