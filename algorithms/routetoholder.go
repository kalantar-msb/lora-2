// Package routetoholder is the SPECIFICATION LAYER for the route-to-holder LoRA
// routing policy. It is not compiling code: it states the decision the simulator
// makes, against the target's real interface names, for /sim2real-translate to
// port. Translate treats it as authoritative for the science.
//
// PINS
//
//   simulation  atantawi/inference-sim c9fb60385bb1bb4ca7b45e5915d2dfe52227b359
//               (tag lora-seams-2026-08-22)
//   target      llm-d/llm-d-router a5cbe600ebade00cf3e9885beaf2bfacddeabce1 (v0.11.0)
//   engine      vllm-project/vllm 33e0602e (v0.7.1-18). Cited only for the meaning
//               of vLLM's LoRA metric and slot policy. It is NOT the deployed vLLM;
//               see D5.
//
// THE DECISION (simulation)
//
// For each request carrying adapter a, the simulator collects the instances whose
// GPU slots hold a ("holders"), in snapshot order (sim/routing.go:277-282). If at
// least one holder exists, it restricts the candidate set to the holders and runs
// the ordinary weighted scorer over that set (sim/routing.go:283-296). If none
// exists, or the request is for the base model, it runs the same weighted scorer
// over every instance (sim/routing.go:299-300). The wrapper adds no score of its
// own (sim/routing.go:375-382). Whenever a holder exists, the chosen instance is a
// holder.
//
// "Holds" is the simulator's per-instance resident set, read with no staleness:
// route-to-holder pins that one snapshot field to Immediate refresh
// (sim/cluster/cluster.go:443-448). Its life cycle under the on-demand creation
// policy and lru eviction, which are the policies this arm runs:
//
//   - A request whose adapter is not resident is a cold miss when it reaches the
//     head of that instance's wait queue (sim/simulator.go:1016-1023). The
//     instance then picks a victim and evicts it, if it is at capacity, at the
//     START of the load (sim/simulator.go:1073-1094). Loads are serialized: one at
//     a time per instance (sim/simulator.go:1051).
//   - The adapter becomes resident only when the load COMPLETES
//     (sim/simulator.go:1165-1191). During the load it is not a holder.
//   - When a request enters the running batch, its adapter is touched to
//     most-recently-used and pinned, once per request (sim/simulator.go:950-959).
//     The pin is released when the request terminates (sim/simulator.go:975-981).
//   - The victim is the least-recently-used UNPINNED adapter
//     (sim/lora/eviction/eviction.go:85-90, over candidates in LRU-to-MRU order,
//     sim/lora/resident_set.go:134-144). If every slot is pinned, no load starts
//     until a pin is released (sim/simulator.go:1075-1083).
//   - On-demand seeds nothing at t=0 and admits every miss
//     (sim/lora/creation/on_demand.go:13-15).
//
// The inner weighted scorer is the simulator's default profile:
// precise-prefix-cache 2, queue-depth 1, kv-utilization 1
// (sim/routing_scorers.go:100-106). Each per-scorer score is clamped to [0,1] and
// summed with weights normalized to 1 (sim/routing.go:191-205,
// sim/routing_scorers.go:145). The highest total
// wins, and exact ties are broken at random (sim/routing.go:207-229).
//
// Every scorer runs over the RESTRICTED set, so the queue-depth and prefix
// min-max ranges are taken over the holders only, not over the fleet
// (sim/routing.go:191-193, given the restricted RouterState from :284-296).
// kv-utilization is absolute, 1 - used/total (sim/routing_scorers.go:269-275).
// The target matches this by construction: filters run before scorers, and the
// scorers see only what the filters returned
// (pkg/epp/scheduling/scheduler_profile.go:160-200). D9 is the one exception.
//
// How long a loading adapter is missing from every holder set is the load
// latency. The simulator charges max(1, ceil(LoadLatency)) ticks
// (sim/simulator.go:1097-1098). The port does not model this: the real load
// supplies it, and the mirror observes its end (D2).
//
// PLACEMENT (candidate)
//
// ONE plugin registration, one Go type, plugin type name "route-to-holder-filter".
// That type implements four interfaces:
//   - scheduling Filter
//     (pkg/epp/framework/interface/scheduling/plugins.go:60-63)
//   - requestcontrol PreRequest
//     (pkg/epp/framework/interface/requestcontrol/plugins.go:70-73)
//   - requestcontrol ResponseHeaderProcessor (same file, :78-81)
//   - requestcontrol ResponseBodyProcessor (same file, :98-101)
// Their method names (Filter, PreRequest, ResponseHeader, ResponseBody) are all
// distinct, so one type can carry all four. A plugin referenced in a scheduling
// profile is registered there under every scheduling interface it implements, and
// every configured plugin is also added to request control under every
// request-control interface it implements:
//   - cmd/epp/runner/runner.go:799
//   - pkg/epp/requestcontrol/request_control_config.go:108-134
// So a single instance both filters and observes. This is a CANDIDATE: the method
// sets have been read, but nothing has been compiled. If it does not hold, the
// eliminations below still do, and the search continues within them.
//
// The weighted half of the decision is NOT reimplemented here. It is the
// profile's existing scorers plus max-score-picker, configured in config.md's
// per-arm settings. The new plugin only narrows the candidate set, as the
// simulator's wrapper does.
//
// Requested of translate: register the type in cmd/epp/runner/runner.go next to
// the other scheduling plugins, as lora-affinity-scorer is at runner.go:618.
//
// ELIMINATIONS (settled; they hold even if the placement above fails)
//
//   - A Scorer cannot carry the decision. Scores are a weighted sum
//     (pkg/epp/scheduling/scheduler_profile.go:236-260), so a holder preference
//     only biases the outcome. The simulator's rule is a hard restriction:
//     whenever a holder exists, the pick is a holder. lora-affinity-scorer is
//     exactly this weaker form, a soft tiered score
//     (pkg/epp/framework/plugins/scheduling/scorer/loraaffinity/lora_affinity.go:84-117).
//   - label-selector-filter cannot carry it. It ignores the request
//     (pkg/epp/framework/plugins/scheduling/filter/bylabel/selector.go:81-91), so
//     it cannot vary by adapter.
//   - WaitingModels is not residency. Its own field comment says vLLM fills it
//     with the same adapters as ActiveModels
//     (pkg/epp/framework/interface/datalayer/metrics.go:29-32).
//   - ActiveModels alone is not residency. It is filled from vLLM's
//     running_lora_adapters label
//     (pkg/epp/framework/plugins/datalayer/extractor/metrics/extractor.go:249-256).
//     At the cited engine that label lists only adapters of RUNNING requests
//     (vllm/engine/llm_engine.py:1628-1634), so an idle resident adapter is
//     absent. It is a subset of the holders. It is used below as a correction,
//     never as the whole set.
//   - The /v1/models listing is not residency either. It lists every registered
//     adapter (vllm/entrypoints/openai/serving_models.py:123-130), including ones
//     evicted from GPU slots. It is a superset of the holders.
//   - No core modification is needed. Filter plus the request-control hooks
//     reach every quantity the decision reads.
//
// DECLARED DEGRADATIONS
//
// The simulator reads residency exactly. The target has no residency signal, so
// this layer keeps a per-pod MIRROR of the simulator's resident set, driven by
// the requests this plugin routed and corrected by ActiveModels. The mirror is
// bridge code; its state machine copies the simulator's (the life cycle above).
// Each place it cannot copy exactly is declared here with its direction of bias.
//
// D1  Traffic this plugin did not route is invisible. That includes other EPP
//     replicas and direct pod traffic. An eviction vLLM made for such traffic is
//     not seen. Bias: the mirror OVERSTATES
//     holding; a request is sent to a "holder" and cold-loads there.
//     Requested of bootstrap: a single EPP replica.
// D2  Load completion is not observable. The mirror marks an adapter resident at
//     the first of two events: (a) a response header from that pod for a request
//     on that adapter; (b) the adapter appearing in that pod's ActiveModels on a
//     metrics refresh. Only a SUCCESSFUL response counts for (a): an error
//     response, such as an adapter that failed to resolve, proves nothing.
//     Both events come after the true completion. For non-streaming responses,
//     (a) arrives only when the response is complete. This bundle's workload
//     declares every client non-streaming; streaming is on only because the
//     measurement's --record-itl forces it (config.md §7). Bias: the mirror
//     UNDERSTATES holding just after a load. A second request in that window
//     sees no holder, falls back to all pods, and may load a second copy.
// D3  The eviction is committed when the plugin dispatches the miss, not when the
//     load starts at the head of the pod's wait queue. Bias: with a non-empty
//     queue, the victim stops counting as a holder EARLIER than in the simulator
//     (understated for the victim). At this bundle's load the queues are expected
//     to be near zero, so the window is small.
// D4  Pin and recency timing.
//     - A pin is taken at dispatch, not at batch admission, and released at end
//       of stream or abort. An adapter whose requests are only queued is
//       therefore protected in the mirror but not in the simulator.
//     - The simulator touches an adapter to most-recently-used only at batch
//       admission (sim/simulator.go:954) and at load completion
//       (sim/lora/resident_set.go:95-109), never at routing time. The mirror
//       also touches it at dispatch, at a warm request's first successful
//       response, and on every applied metrics refresh that lists it as
//       running. Ordering by "last seen running" is closer to vLLM, which
//       touches on every activation (vllm/lora/models.py:713 at the cited
//       engine), than to the simulator.
//     - The simulator's Pin is a no-op unless the adapter is resident
//       (sim/lora/resident_set.go:160-163). The mirror also pins an adapter whose
//       load is still pending.
//     Bias: the mirror picks a different victim when queued or long-running work
//     exists, favouring the adapters with such work. Like D3, this is small at
//     low queue depth.
// D5  The mirror assumes vLLM keeps GPU slots as an LRU list
//     (vllm/lora/models.py:704-710 at the cited engine) and that
//     running_lora_adapters lists running requests only (cited above). Both are
//     cited against v0.7.1, not the deployed image; config.md marks them CONFIRM.
//     If the deployed vLLM differs, the mirror's victim choice diverges. Bias:
//     unknown in sign.
// D6  Slot count comes from MaxActiveModels (vLLM's max_lora;
//     pkg/epp/framework/interface/datalayer/metrics.go:34). vLLM emits the LoRA
//     metric family only once an adapter has been loaded
//     (pkg/epp/framework/plugins/datalayer/extractor/metrics/loraspec.go:53-57),
//     so before that it is zero. FallbackSlots is used until it appears. Bias:
//     none if FallbackSlots equals --max-loras.
// D7  The inner scorers differ from the simulator's. The stock profile weights
//     are queue 2, kv-cache 2, prefix-cache 3
//     (pkg/epp/config/loader/defaults.go:49-51), against the simulator's
//     queue 1, kv 1, precise-prefix 2. Normalized: prefix 0.43 against 0.50,
//     queue and kv 0.29 against 0.25. The target's prefix-cache-scorer is an
//     EPP-side approximate index; the simulator's reads the instance cache. They
//     are kept at the stock values so this arm differs from baseline only by the
//     filter. Bias: none when prefix scores tie, which is expected for this
//     bundle's synthetic prompts. Otherwise the target weights the prefix term
//     slightly less.
// D8  Ties among equal scores rotate round-robin in max-score-picker
//     (pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go:105-118).
//     The simulator breaks them at random (sim/routing.go:226-229). The published
//     lora-control leaf turned the simulator's deterministic tie-break ON. Bias:
//     the target spreads ties more evenly than either.
// D9  The utilization saturation detector is appended as a filter after this
//     plugin in every profile (pkg/epp/config/loader/defaults.go:276-323). It
//     drops overloaded pods but never returns an empty set
//     (pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization/detector.go:207-232),
//     so it narrows WITHIN the holders and cannot break the holder rule. The
//     simulator has no such stage. Baseline shares it. Bias: under saturation the
//     target avoids an overloaded holder when another holder exists.
// D10 Whether a request is for the base model comes from a parameter
//     (BaseModels), not from the request. The simulator uses the empty adapter
//     id. If a base-model name is missing from the list, its requests enter the
//     mirror as an "adapter". Bias: restricted routing for base-model traffic.
//     That traffic does not exist in this bundle's workload.
// D11 Freshness. The simulator reads residency with no staleness. The mirror is
//     updated on request events (immediate). Its ActiveModels correction is
//     applied from Filter only, for the candidate pods, at most once per metrics
//     refresh, and sampled when a request arrives: a refresh that falls between
//     two requests is never seen. Same direction as D2.
// D12 Over capacity. When every slot is pinned, the simulator starts no load
//     (sim/simulator.go:1075-1083), and its Store refuses rather than exceed
//     capacity (sim/lora/resident_set.go:100-102). The mirror cannot refuse a
//     confirmation the pod has already given. It stores anyway, so a pod's list
//     can briefly run over its slot count until the next eviction trims it.
//     Bias: the mirror OVERSTATES holding during that interval.
// D13 Aborted loads. A request aborted before any successful response leaves no
//     confirmation. Its pending load is cleared when it ends, unless another
//     unconfirmed request for the same adapter on the same pod is still in
//     flight. The adapter may in fact have loaded. If so, an ActiveModels
//     correction restores it, but only once a later request for that adapter
//     is running on that pod when Filter samples the metrics (D11). Bias: the
//     mirror UNDERSTATES holding until then.
package routetoholder

import (
	"context"
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Parameters, decoded strictly from the plugin's `parameters` JSON.
type Parameters struct {
	// FallbackSlots is the GPU adapter slot count used while a pod's
	// MaxActiveModels is still zero (D6). Must be > 0. Set it to --max-loras.
	FallbackSlots int `json:"fallbackSlots"`
	// BaseModels lists TargetModel values that are the base model, not an
	// adapter. Requests for them skip the filter (D10).
	BaseModels []string `json:"baseModels"`
}

// RouteToHolder is the one registered type (see PLACEMENT).
//
// Concurrency: Filter runs on the request path, while PreRequest, ResponseHeader
// and ResponseBody run on other goroutines. All three mutate the state Filter
// reads, so every access holds mu.
type RouteToHolder struct {
	params Parameters

	mu       sync.Mutex
	pods     map[fwkdl.ID]*podMirror
	inflight map[string]*inflightReq // keyed by InferenceRequest.RequestID
}

// podMirror copies the simulator's residentSet
// (sim/lora/resident_set.go:15-31): a capacity-bounded LRU of adapter ids with
// reference-counted pins, plus the one in-progress load the simulator serializes.
type podMirror struct {
	lru         []string       // resident adapters, LRU first, MRU last
	pins        map[string]int // in-flight references per adapter
	loading     map[string]bool
	slots       int   // last non-zero MaxActiveModels seen for this pod (D6)
	lastScrape  int64 // UpdateTime (ns) of the last ActiveModels correction applied
}

type inflightReq struct {
	pod       fwkdl.ID
	adapter   string
	confirmed bool // a response header has been seen
}

// ---------------------------------------------------------------- the decision

// Filter is the simulator's RouteToHolder.Route (sim/routing.go:263-301) with the
// inner weighted policy removed: the profile's scorers and picker run on
// whatever this returns.
func (p *RouteToHolder) Filter(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	adapter := req.TargetModel // pkg/epp/framework/interface/scheduling/types.go:49-50
	if p.isBaseModel(adapter) {
		return endpoints // sim/routing.go:299-300: base-model request -> unconstrained
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	holders := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints { // input order, as sim/routing.go:277-282
		m := p.mirrorFor(ep.GetMetadata().ID)
		p.applyScrape(m, ep.GetMetrics())
		if m.isResident(adapter) {
			holders = append(holders, ep)
		}
	}
	if len(holders) > 0 {
		return holders // sim/routing.go:283-296
	}
	return endpoints // sim/routing.go:299-300: no holder -> every pod
}

// --------------------------------------------- mirror updates (bridge; D1-D6)

// PreRequest runs once the pod is chosen. It copies, at dispatch, the
// simulator's cold-miss eviction (D3) and batch-admission touch and pin (D4).
func (p *RouteToHolder) PreRequest(ctx context.Context, req *fwksched.InferenceRequest, res *fwksched.SchedulingResult) error {
	adapter := req.TargetModel
	if p.isBaseModel(adapter) {
		return nil
	}
	pod, ok := primaryTarget(res)
	if !ok {
		return nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	m := p.mirrorFor(pod)

	if m.isResident(adapter) {
		m.touch(adapter) // sim/simulator.go:954
	} else if !m.loading[adapter] {
		// Cold miss: reserve a slot by evicting now, as at load start
		// (sim/simulator.go:1073-1094). Resident plus loading counts as occupied,
		// because the simulator's reservation keeps the slot free for the load.
		if len(m.lru)+len(m.loading) >= p.slotsOf(m) {
			if victim, ok := m.lruUnpinned(); ok { // sim/lora/eviction/eviction.go:85-90
				m.evict(victim)
			}
			// Every slot pinned: the simulator starts no load and waits
			// (sim/simulator.go:1075-1083). The mirror records the pending load
			// anyway; confirmation stores it once a pin is released.
		}
		m.loading[adapter] = true
	}
	m.pins[adapter]++ // sim/simulator.go:955-958, taken once per request (D4)
	p.inflight[req.RequestID] = &inflightReq{pod: pod, adapter: adapter}
	return nil
}

// ResponseHeader is confirmation (a) of D2: the pod answered, so its load is done.
func (p *RouteToHolder) ResponseHeader(ctx context.Context, req *fwksched.InferenceRequest, resp *fwkrc.Response, target *fwkdl.EndpointMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	r, ok := p.inflight[req.RequestID]
	if !ok || r.confirmed || !servedOK(resp) {
		return // an error response proves nothing about residency (D2)
	}
	r.confirmed = true
	p.store(p.mirrorFor(r.pod), r.adapter)
}

// ResponseBody releases the pin at end of stream, the simulator's terminal
// release (sim/simulator.go:975-981). The director also calls it on abort for any
// request that picked a pod
// (pkg/epp/framework/interface/requestcontrol/plugins.go:63-69), so pins do not
// leak.
func (p *RouteToHolder) ResponseBody(ctx context.Context, req *fwksched.InferenceRequest, resp *fwkrc.Response, target *fwkdl.EndpointMetadata) {
	if !endOfStream(resp) {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	r, ok := p.inflight[req.RequestID]
	if !ok {
		return
	}
	delete(p.inflight, req.RequestID)
	m := p.mirrorFor(r.pod)
	if !r.confirmed {
		if endedWithResponse(resp) && servedOK(resp) {
			p.store(m, r.adapter) // non-streaming: the body is the first answer (D2)
		} else if !p.otherPending(r.pod, r.adapter) {
			delete(m.loading, r.adapter) // aborted before confirmation (D13)
		}
	}
	if m.pins[r.adapter] > 0 {
		m.pins[r.adapter]--
	}
	if m.pins[r.adapter] == 0 {
		delete(m.pins, r.adapter)
	}
}

// applyScrape is confirmation (b) of D2. Any adapter the pod reports as running
// is resident there (D5), so it is stored and touched. It is applied at most
// once per metrics refresh, when Filter runs (D11). Nothing is ever removed on a scrape, because absence from
// ActiveModels means idle, not evicted.
func (p *RouteToHolder) applyScrape(m *podMirror, met *fwkdl.Metrics) {
	if met == nil || met.UpdateTime.UnixNano() == m.lastScrape {
		return
	}
	m.lastScrape = met.UpdateTime.UnixNano()
	if met.MaxActiveModels > 0 { // pkg/epp/framework/interface/datalayer/metrics.go:34
		m.slots = met.MaxActiveModels
	}
	for adapter := range met.ActiveModels { // pkg/epp/framework/plugins/datalayer/extractor/metrics/extractor.go:249-256
		if p.isBaseModel(adapter) {
			continue
		}
		if m.isResident(adapter) {
			m.touch(adapter)
			continue
		}
		p.storeAt(m, p.slotsOf(m), adapter)
	}
}

// store copies completeAdapterLoad (sim/simulator.go:1165-1191): the adapter
// becomes resident at MRU and its pending load clears.
func (p *RouteToHolder) store(m *podMirror, adapter string) {
	p.storeAt(m, p.slotsOf(m), adapter)
}

// storeAt is residentSet.Store (sim/lora/resident_set.go:95-109). If the adapter
// is already resident it is touched. Otherwise, at capacity, the least-recently
// used unpinned adapter is evicted first. If every slot is pinned, the simulator's
// Store refuses; the mirror stores anyway and lets the list run over capacity.
// The pod proved the adapter is resident, and the excess is corrected by the
// next eviction.
func (p *RouteToHolder) storeAt(m *podMirror, slots int, adapter string) {
	delete(m.loading, adapter)
	if m.isResident(adapter) {
		m.touch(adapter)
		return
	}
	for len(m.lru) >= slots {
		victim, ok := m.lruUnpinned()
		if !ok {
			break
		}
		m.evict(victim)
	}
	m.lru = append(m.lru, adapter)
}

// ------------------------------------------------------ resident-set helpers

// isResident is residentSet.IsResident (sim/lora/resident_set.go:74-77). A
// loading adapter is NOT resident, as in the simulator.
func (m *podMirror) isResident(a string) bool {
	for _, id := range m.lru {
		if id == a {
			return true
		}
	}
	return false
}

// touch is residentSet.Touch (sim/lora/resident_set.go:80-87): move to MRU, no-op
// if absent.
func (m *podMirror) touch(a string) {
	for i, id := range m.lru {
		if id == a {
			m.lru = append(append(m.lru[:i:i], m.lru[i+1:]...), a)
			return
		}
	}
}

// lruUnpinned is the lru eviction policy (sim/lora/eviction/eviction.go:85-90):
// the first unpinned id in LRU-to-MRU order (sim/lora/resident_set.go:134-144).
func (m *podMirror) lruUnpinned() (string, bool) {
	for _, id := range m.lru {
		if m.pins[id] == 0 {
			return id, true
		}
	}
	return "", false
}

// evict is residentSet.Evict (sim/lora/resident_set.go:148-156): removes an
// unpinned id.
func (m *podMirror) evict(a string) {
	for i, id := range m.lru {
		if id == a && m.pins[a] == 0 {
			m.lru = append(m.lru[:i:i], m.lru[i+1:]...)
			return
		}
	}
}

// slotsOf is the pod's slot count: the cached MaxActiveModels, else FallbackSlots
// (D6).
func (p *RouteToHolder) slotsOf(m *podMirror) int {
	if m.slots > 0 {
		return m.slots
	}
	return p.params.FallbackSlots
}

// mirrorFor returns the pod's mirror, creating an empty one for a pod not seen
// before. An empty mirror is the simulator's on-demand start state
// (sim/lora/creation/on_demand.go:13).
func (p *RouteToHolder) mirrorFor(id fwkdl.ID) *podMirror {
	m, ok := p.pods[id]
	if !ok {
		m = &podMirror{pins: map[string]int{}, loading: map[string]bool{}}
		p.pods[id] = m
	}
	return m
}

// otherPending reports whether another in-flight, unconfirmed request is waiting
// on the same adapter's load on the same pod. The caller has already removed its
// own entry from inflight.
func (p *RouteToHolder) otherPending(pod fwkdl.ID, adapter string) bool {
	for _, r := range p.inflight {
		if r.pod == pod && r.adapter == adapter && !r.confirmed {
			return true
		}
	}
	return false
}

func (p *RouteToHolder) isBaseModel(model string) bool {
	for _, b := range p.params.BaseModels {
		if b == model {
			return true
		}
	}
	return false
}

// --------------------------------------------- target-API adapters (unknowns)
//
// Each of these needs one accessor confirmed against the pinned checkout. They
// are left without bodies on purpose, so that a wrong guess fails to compile
// rather than mis-scoring.

// primaryTarget returns the ID of the endpoint chosen by the primary profile:
// res.ProfileResults[res.PrimaryProfileName].TargetEndpoints[0]
// (pkg/epp/framework/interface/scheduling/types.go:167-180). Confirm the primary
// profile is always populated when PreRequest runs, and that ID is the same key
// Filter sees through GetMetadata().ID.
func primaryTarget(res *fwksched.SchedulingResult) (fwkdl.ID, bool)

// endOfStream reports response.EndOfStream
// (pkg/epp/framework/interface/requestcontrol/plugins.go:83-97).
func endOfStream(resp *fwkrc.Response) bool

// servedOK reports whether the response carries a successful (2xx) status.
// Confirm where Response exposes the HTTP status. A candidate is the "status"
// entry of Response.Headers, which the EPP's own server inspects
// (pkg/epp/handlers/server.go:574). servedOK is also called from ResponseBody. A
// comment in the target says headers are empty during body processing, but the
// director passes Response.Headers to both hooks (pkg/epp/requestcontrol/director.go:550-553).
// Confirm which is true at the pinned ref. Response headers reach plugins
// for every response, with no status check
// (pkg/epp/handlers/response.go:135-139).
func servedOK(resp *fwkrc.Response) bool

// endedWithResponse reports whether the final call carries a served response
// rather than an abort. Confirm against Response.TerminationCause, which the
// interface comment says does not distinguish success from an error response.
// If it cannot be told apart, return false: the ActiveModels correction then
// supplies confirmation (b).
func endedWithResponse(resp *fwkrc.Response) bool
