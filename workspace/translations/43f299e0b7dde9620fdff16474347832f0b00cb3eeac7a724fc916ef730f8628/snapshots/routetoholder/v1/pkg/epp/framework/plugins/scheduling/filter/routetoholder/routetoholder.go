/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package routetoholder provides the route-to-holder LoRA filter.
//
// For a request carrying adapter a, the filter keeps only the endpoints whose GPU
// adapter slots hold a. If no endpoint holds it, or the request is for the base
// model, every endpoint is kept. The profile's scorers and picker then run over
// what the filter returns, so whenever a holder exists the pick is a holder.
//
// The EPP has no residency signal, so the plugin keeps a per-endpoint mirror of
// the resident set: a capacity-bounded LRU with reference-counted pins, updated
// from the requests this plugin routed (PreRequest, ResponseHeader, ResponseBody)
// and corrected by each endpoint's ActiveModels metric. The mirror only sees
// traffic routed by this EPP replica, so run a single replica.
//
// Ported from BLIS's route-to-holder policy (atantawi/inference-sim c9fb6038,
// sim/routing.go:263-301). The specification layer, with its declared
// degradations D1-D13, is algorithms/routetoholder.go in the sim2real bundle.
package routetoholder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// RouteToHolderType is the type of the route-to-holder filter.
const RouteToHolderType = "route-to-holder-filter"

// Parameters configures the route-to-holder filter.
type Parameters struct {
	// FallbackSlots is the GPU adapter slot count used while an endpoint's
	// MaxActiveModels is still zero (vLLM emits it only after the first adapter
	// load). Must be > 0. Set it to vLLM's --max-loras.
	FallbackSlots int `json:"fallbackSlots"`
	// BaseModels lists TargetModel values that are the base model, not an
	// adapter. Requests for them are not filtered and do not enter the mirror.
	BaseModels []string `json:"baseModels"`
}

func (p Parameters) validate() error {
	if p.FallbackSlots <= 0 {
		return fmt.Errorf("fallbackSlots must be > 0, got %d", p.FallbackSlots)
	}
	return nil
}

// compile-time type assertion
var _ fwksched.Filter = &RouteToHolder{}
var _ fwkrc.PreRequest = &RouteToHolder{}
var _ fwkrc.ResponseHeaderProcessor = &RouteToHolder{}
var _ fwkrc.ResponseBodyProcessor = &RouteToHolder{}

// Factory defines the factory function for the route-to-holder filter.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	if rawParameters == nil {
		return nil, fmt.Errorf("the '%s' filter requires parameters (fallbackSlots)", RouteToHolderType)
	}
	params := Parameters{}
	if err := rawParameters.Decode(&params); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", RouteToHolderType, err)
	}
	return New(name, params)
}

// New returns a route-to-holder filter with the given parameters.
func New(name string, params Parameters) (*RouteToHolder, error) {
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' filter - %w", RouteToHolderType, err)
	}
	base := make(map[string]bool, len(params.BaseModels))
	for _, m := range params.BaseModels {
		base[m] = true
	}
	return &RouteToHolder{
		typedName:     plugin.TypedName{Type: RouteToHolderType, Name: name},
		fallbackSlots: params.FallbackSlots,
		baseModels:    base,
		pods:          map[fwkdl.ID]*podMirror{},
		inflight:      map[string]*inflightReq{},
	}, nil
}

// RouteToHolder restricts the candidate set to the endpoints holding the
// request's adapter, and keeps the residency mirror that decision reads.
//
// Filter runs on the request path while PreRequest, ResponseHeader and
// ResponseBody run on other goroutines and mutate the mirror, so every access to
// pods and inflight holds mu.
type RouteToHolder struct {
	typedName     plugin.TypedName
	fallbackSlots int
	baseModels    map[string]bool

	mu       sync.Mutex
	pods     map[fwkdl.ID]*podMirror
	inflight map[string]*inflightReq // keyed by InferenceRequest.RequestID
}

// podMirror copies the simulator's residentSet (sim/lora/resident_set.go:15-31).
type podMirror struct {
	lru        []string       // resident adapters, LRU first, MRU last
	pins       map[string]int // in-flight references per adapter
	loading    map[string]bool
	slots      int   // last non-zero MaxActiveModels seen for this endpoint
	lastScrape int64 // UpdateTime (ns) of the last ActiveModels correction applied
}

type inflightReq struct {
	pod       fwkdl.ID
	adapter   string
	confirmed bool // a successful response has been seen
}

// TypedName returns the typed name of the plugin.
func (p *RouteToHolder) TypedName() plugin.TypedName {
	return p.typedName
}

// Filter returns the endpoints whose mirror holds the request's adapter, in input
// order (sim/routing.go:277-296). If none holds it, or the request is for the
// base model, it returns every endpoint (sim/routing.go:299-300).
func (p *RouteToHolder) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	if request == nil || p.isBaseModel(request.TargetModel) {
		return endpoints
	}
	adapter := request.TargetModel

	p.mu.Lock()
	defer p.mu.Unlock()

	holders := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		md := ep.GetMetadata()
		if md == nil {
			continue
		}
		m := p.mirrorFor(md.ID)
		p.applyScrape(m, ep.GetMetrics())
		if m.isResident(adapter) {
			holders = append(holders, ep)
		}
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("Filtered endpoints by adapter residency",
		"adapter", adapter, "candidates", len(endpoints), "holders", len(holders))

	if len(holders) > 0 {
		return holders
	}
	return endpoints
}

// PreRequest runs once the endpoint is chosen. On a cold miss it evicts the
// least-recently-used unpinned adapter if the endpoint is at capacity and marks
// the adapter loading (sim/simulator.go:1073-1094). It then pins the adapter for
// the life of the request (sim/simulator.go:950-959).
func (p *RouteToHolder) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	if request == nil || request.RequestID == "" || p.isBaseModel(request.TargetModel) {
		return nil
	}
	pod, ok := primaryTarget(schedulingResult)
	if !ok {
		return nil
	}
	adapter := request.TargetModel

	p.mu.Lock()
	defer p.mu.Unlock()

	// A request id seen twice releases its earlier pin, so it cannot leak.
	if prev, ok := p.inflight[request.RequestID]; ok {
		p.release(prev)
		delete(p.inflight, request.RequestID)
	}

	m := p.mirrorFor(pod)
	if m.isResident(adapter) {
		m.touch(adapter)
	} else if !m.loading[adapter] {
		// Resident plus loading counts as occupied: the simulator keeps the
		// reserved slot free for the load. If every slot is pinned no victim
		// exists; the pending load is recorded anyway and stored on confirmation.
		if len(m.lru)+len(m.loading) >= p.slotsOf(m) {
			if victim, ok := m.lruUnpinned(); ok {
				m.evict(victim)
			}
		}
		m.loading[adapter] = true
	}
	m.pins[adapter]++
	p.inflight[request.RequestID] = &inflightReq{pod: pod, adapter: adapter}

	log.FromContext(ctx).V(logutil.TRACE).Info("Route-to-holder dispatch",
		"adapter", adapter, "endpoint", pod.String(), "resident", m.lru, "loading", len(m.loading))
	return nil
}

// ResponseHeader confirms the load: a successful response from the endpoint
// proves the adapter is resident there (sim/simulator.go:1165-1191). An error
// response proves nothing.
func (p *RouteToHolder) ResponseHeader(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || !servedOK(response) {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	r, ok := p.inflight[request.RequestID]
	if !ok || r.confirmed {
		return
	}
	r.confirmed = true
	p.store(p.mirrorFor(r.pod), r.adapter)
}

// ResponseBody releases the pin at end of stream (sim/simulator.go:975-981). The
// director also calls it with EndOfStream on abort for any request that picked
// an endpoint, so pins do not leak. A request that ends unconfirmed either
// confirms here (a complete successful response) or, if aborted, clears its
// pending load unless another unconfirmed request is waiting on it.
func (p *RouteToHolder) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	r, ok := p.inflight[request.RequestID]
	if !ok {
		return
	}
	delete(p.inflight, request.RequestID)
	m := p.mirrorFor(r.pod)
	if !r.confirmed {
		if response.TerminationCause == fwkrc.TerminationCauseNatural && servedOK(response) {
			p.store(m, r.adapter)
		} else if !p.otherPending(r.pod, r.adapter) {
			delete(m.loading, r.adapter)
		}
	}
	p.release(r)
}

// release drops one pin taken by r.
func (p *RouteToHolder) release(r *inflightReq) {
	m := p.mirrorFor(r.pod)
	if m.pins[r.adapter] > 0 {
		m.pins[r.adapter]--
	}
	if m.pins[r.adapter] == 0 {
		delete(m.pins, r.adapter)
	}
}

// applyScrape stores and touches every adapter the endpoint reports in
// ActiveModels: a running adapter is resident. It runs at most once per metrics
// refresh. Nothing is removed, since absence from ActiveModels means idle, not
// evicted.
func (p *RouteToHolder) applyScrape(m *podMirror, met *fwkdl.Metrics) {
	if met == nil || met.UpdateTime.UnixNano() == m.lastScrape {
		return
	}
	m.lastScrape = met.UpdateTime.UnixNano()
	if met.MaxActiveModels > 0 {
		m.slots = met.MaxActiveModels
	}
	for adapter := range met.ActiveModels {
		if p.isBaseModel(adapter) {
			continue
		}
		p.store(m, adapter)
	}
}

// store is residentSet.Store (sim/lora/resident_set.go:95-109) plus the end of
// the pending load. A resident adapter is touched. Otherwise, at capacity, the
// least-recently-used unpinned adapters are evicted first. If every slot is
// pinned the simulator refuses; the mirror stores anyway, because the endpoint
// has already proved residency, and the next eviction trims the excess.
func (p *RouteToHolder) store(m *podMirror, adapter string) {
	delete(m.loading, adapter)
	if m.isResident(adapter) {
		m.touch(adapter)
		return
	}
	for len(m.lru) >= p.slotsOf(m) {
		victim, ok := m.lruUnpinned()
		if !ok {
			break
		}
		m.evict(victim)
	}
	m.lru = append(m.lru, adapter)
}

// isResident is residentSet.IsResident: a loading adapter is not resident.
func (m *podMirror) isResident(a string) bool {
	for _, id := range m.lru {
		if id == a {
			return true
		}
	}
	return false
}

// touch moves a resident adapter to MRU; no-op if absent.
func (m *podMirror) touch(a string) {
	for i, id := range m.lru {
		if id == a {
			m.lru = append(append(m.lru[:i:i], m.lru[i+1:]...), a)
			return
		}
	}
}

// lruUnpinned returns the first unpinned adapter in LRU-to-MRU order
// (sim/lora/eviction/eviction.go:85-90).
func (m *podMirror) lruUnpinned() (string, bool) {
	for _, id := range m.lru {
		if m.pins[id] == 0 {
			return id, true
		}
	}
	return "", false
}

// evict removes an unpinned adapter.
func (m *podMirror) evict(a string) {
	for i, id := range m.lru {
		if id == a && m.pins[a] == 0 {
			m.lru = append(m.lru[:i:i], m.lru[i+1:]...)
			return
		}
	}
}

// slotsOf is the endpoint's slot count: the cached MaxActiveModels, else
// FallbackSlots.
func (p *RouteToHolder) slotsOf(m *podMirror) int {
	if m.slots > 0 {
		return m.slots
	}
	return p.fallbackSlots
}

// mirrorFor returns the endpoint's mirror, creating an empty one (the
// simulator's on-demand start state) for an endpoint not seen before.
func (p *RouteToHolder) mirrorFor(id fwkdl.ID) *podMirror {
	m, ok := p.pods[id]
	if !ok {
		m = &podMirror{pins: map[string]int{}, loading: map[string]bool{}}
		p.pods[id] = m
	}
	return m
}

// otherPending reports whether another in-flight, unconfirmed request is waiting
// on the same adapter's load on the same endpoint. The caller has already
// removed its own entry from inflight.
func (p *RouteToHolder) otherPending(pod fwkdl.ID, adapter string) bool {
	for _, r := range p.inflight {
		if r.pod == pod && r.adapter == adapter && !r.confirmed {
			return true
		}
	}
	return false
}

func (p *RouteToHolder) isBaseModel(model string) bool {
	return p.baseModels[model]
}

// primaryTarget returns the ID of the endpoint the primary profile selected.
func primaryTarget(result *fwksched.SchedulingResult) (fwkdl.ID, bool) {
	if result == nil {
		return fwkdl.ID{}, false
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 || primary.TargetEndpoints[0] == nil {
		return fwkdl.ID{}, false
	}
	md := primary.TargetEndpoints[0].GetMetadata()
	if md == nil {
		return fwkdl.ID{}, false
	}
	return md.ID, true
}

var errNoStatus = errors.New("no status header")

// responseStatus returns the HTTP status from the response headers. Envoy may
// deliver the pseudo-header as ":status"; the EPP's own server reads "status"
// (pkg/epp/handlers/server.go), so both are checked.
func responseStatus(response *fwkrc.Response) (int, error) {
	if response == nil {
		return 0, errNoStatus
	}
	for _, key := range []string{":status", "status"} {
		if v, ok := response.Headers[key]; ok {
			return strconv.Atoi(v)
		}
	}
	return 0, errNoStatus
}

// servedOK reports whether the response carries a 2xx status.
func servedOK(response *fwkrc.Response) bool {
	code, err := responseStatus(response)
	return err == nil && code >= 200 && code < 300
}
