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

package routetoholder

import (
	"context"
	"encoding/json"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

const (
	baseModel    = "ibm-granite/granite-3.3-2b-instruct"
	citations    = "citations"
	uncertainty  = "uncertainty"
	statusHeader = ":status"
)

func newFilter(t *testing.T) *RouteToHolder {
	t.Helper()
	f, err := New("test", Parameters{FallbackSlots: 3, BaseModels: []string{baseModel}})
	require.NoError(t, err)
	return f
}

func endpoint(name string, met *fwkdl.Metrics) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, met, nil)
}

func names(endpoints []fwksched.Endpoint) []string {
	out := make([]string, len(endpoints))
	for i, e := range endpoints {
		out[i] = e.GetMetadata().ID.Name
	}
	return out
}

func resultFor(ep fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{ep}},
		},
	}
}

func req(id, model string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{RequestID: id, TargetModel: model}
}

func ok200() *fwkrc.Response {
	return &fwkrc.Response{Headers: map[string]string{statusHeader: "200"}}
}

func endOK() *fwkrc.Response {
	return &fwkrc.Response{
		Headers:          map[string]string{statusHeader: "200"},
		EndOfStream:      true,
		TerminationCause: fwkrc.TerminationCauseNatural,
	}
}

func abort() *fwkrc.Response {
	return &fwkrc.Response{Headers: map[string]string{}, EndOfStream: true, TerminationCause: fwkrc.TerminationCauseClientDisconnect}
}

// dispatch runs PreRequest then a successful ResponseHeader, leaving the request in flight.
func dispatch(t *testing.T, f *RouteToHolder, id, adapter string, ep fwksched.Endpoint) {
	t.Helper()
	ctx := context.Background()
	require.NoError(t, f.PreRequest(ctx, req(id, adapter), resultFor(ep)))
	f.ResponseHeader(ctx, req(id, adapter), ok200(), ep.GetMetadata())
}

// complete runs dispatch then end of stream, releasing the pin.
func complete(t *testing.T, f *RouteToHolder, id, adapter string, ep fwksched.Endpoint) {
	t.Helper()
	dispatch(t, f, id, adapter, ep)
	f.ResponseBody(context.Background(), req(id, adapter), endOK(), ep.GetMetadata())
}

func resident(f *RouteToHolder, ep fwksched.Endpoint) []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string{}, f.mirrorFor(ep.GetMetadata().ID).lru...)
}

func TestFactory(t *testing.T) {
	handle := utils.NewTestHandle(utils.NewTestContext(t))
	tests := []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{name: "valid", raw: `{"fallbackSlots": 3, "baseModels": ["` + baseModel + `"]}`},
		{name: "missing parameters", raw: ``, wantErr: true},
		{name: "zero slots", raw: `{"fallbackSlots": 0}`, wantErr: true},
		{name: "unknown field", raw: `{"fallbackSlots": 3, "bogus": 1}`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := Factory("rth", plugin.StrictDecoder(json.RawMessage(tt.raw)), handle)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, plugin.TypedName{Type: RouteToHolderType, Name: "rth"}, p.TypedName())
			f := p.(*RouteToHolder)
			assert.Equal(t, 3, f.fallbackSlots)
			assert.True(t, f.isBaseModel(baseModel))
		})
	}
}

func TestFilterNoHolderReturnsAll(t *testing.T) {
	f := newFilter(t)
	eps := []fwksched.Endpoint{endpoint("a", &fwkdl.Metrics{}), endpoint("b", &fwkdl.Metrics{})}
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), req("r", citations), eps)))
}

func TestFilterRestrictsToHolders(t *testing.T) {
	f := newFilter(t)
	a, b, c := endpoint("a", &fwkdl.Metrics{}), endpoint("b", &fwkdl.Metrics{}), endpoint("c", &fwkdl.Metrics{})
	complete(t, f, "1", citations, c)
	complete(t, f, "2", citations, a)
	complete(t, f, "3", uncertainty, b)

	// Input order is preserved.
	got := f.Filter(context.Background(), req("r", citations), []fwksched.Endpoint{a, b, c})
	assert.Equal(t, []string{"a", "c"}, names(got))
	got = f.Filter(context.Background(), req("r", uncertainty), []fwksched.Endpoint{a, b, c})
	assert.Equal(t, []string{"b"}, names(got))
}

func TestFilterBaseModelUnconstrained(t *testing.T) {
	f := newFilter(t)
	a, b := endpoint("a", &fwkdl.Metrics{}), endpoint("b", &fwkdl.Metrics{})
	complete(t, f, "1", baseModel, a)
	assert.Empty(t, resident(f, a), "base model must not enter the mirror")
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), req("r", baseModel), []fwksched.Endpoint{a, b})))
}

func TestLoadingIsNotResidentUntilConfirmed(t *testing.T) {
	f := newFilter(t)
	ctx := context.Background()
	a, b := endpoint("a", &fwkdl.Metrics{}), endpoint("b", &fwkdl.Metrics{})
	require.NoError(t, f.PreRequest(ctx, req("1", citations), resultFor(a)))
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(ctx, req("x", citations), []fwksched.Endpoint{a, b})))

	// An error response proves nothing.
	f.ResponseHeader(ctx, req("1", citations), &fwkrc.Response{Headers: map[string]string{statusHeader: "404"}}, a.GetMetadata())
	assert.Empty(t, resident(f, a))

	f.ResponseHeader(ctx, req("1", citations), &fwkrc.Response{Headers: map[string]string{"status": "200"}}, a.GetMetadata())
	assert.Equal(t, []string{citations}, resident(f, a))
	assert.Equal(t, []string{"a"}, names(f.Filter(ctx, req("x", citations), []fwksched.Endpoint{a, b})))
}

func TestBodyConfirmsWhenNoHeaderConfirmation(t *testing.T) {
	f := newFilter(t)
	ctx := context.Background()
	a := endpoint("a", &fwkdl.Metrics{})
	require.NoError(t, f.PreRequest(ctx, req("1", citations), resultFor(a)))
	f.ResponseBody(ctx, req("1", citations), endOK(), a.GetMetadata())
	assert.Equal(t, []string{citations}, resident(f, a))
}

func TestAbortClearsPendingLoad(t *testing.T) {
	f := newFilter(t)
	ctx := context.Background()
	a := endpoint("a", &fwkdl.Metrics{})
	require.NoError(t, f.PreRequest(ctx, req("1", citations), resultFor(a)))
	require.NoError(t, f.PreRequest(ctx, req("2", citations), resultFor(a)))

	f.ResponseBody(ctx, req("1", citations), abort(), a.GetMetadata())
	f.mu.Lock()
	assert.True(t, f.pods[a.GetMetadata().ID].loading[citations], "another request still waits on the load")
	f.mu.Unlock()

	f.ResponseBody(ctx, req("2", citations), abort(), a.GetMetadata())
	f.mu.Lock()
	m := f.pods[a.GetMetadata().ID]
	assert.False(t, m.loading[citations])
	assert.Empty(t, m.pins)
	f.mu.Unlock()
	assert.Empty(t, f.inflight)
	assert.Empty(t, resident(f, a))
}

func TestColdMissEvictsLRUUnpinned(t *testing.T) {
	f := newFilter(t) // fallbackSlots 3
	a := endpoint("a", &fwkdl.Metrics{})
	complete(t, f, "1", "x", a)
	complete(t, f, "2", "y", a)
	complete(t, f, "3", "z", a)
	assert.Equal(t, []string{"x", "y", "z"}, resident(f, a))

	// A warm hit touches x to MRU, so y is now LRU.
	complete(t, f, "4", "x", a)
	assert.Equal(t, []string{"y", "z", "x"}, resident(f, a))

	// The cold miss evicts y at dispatch, before the load confirms.
	require.NoError(t, f.PreRequest(context.Background(), req("5", "w"), resultFor(a)))
	assert.Equal(t, []string{"z", "x"}, resident(f, a))
	f.ResponseHeader(context.Background(), req("5", "w"), ok200(), a.GetMetadata())
	assert.Equal(t, []string{"z", "x", "w"}, resident(f, a))
}

func TestPinnedAdapterIsNotEvicted(t *testing.T) {
	f := newFilter(t)
	a := endpoint("a", &fwkdl.Metrics{})
	dispatch(t, f, "1", "x", a) // stays in flight, x pinned
	complete(t, f, "2", "y", a)
	complete(t, f, "3", "z", a)

	require.NoError(t, f.PreRequest(context.Background(), req("4", "w"), resultFor(a)))
	assert.Equal(t, []string{"x", "z"}, resident(f, a), "x is LRU but pinned; y is evicted")
}

func TestAllPinnedOverCapacity(t *testing.T) {
	f := newFilter(t)
	a := endpoint("a", &fwkdl.Metrics{})
	dispatch(t, f, "1", "x", a)
	dispatch(t, f, "2", "y", a)
	dispatch(t, f, "3", "z", a)
	dispatch(t, f, "4", "w", a)
	assert.Equal(t, []string{"x", "y", "z", "w"}, resident(f, a), "confirmation stores beyond capacity when all slots are pinned")

	f.ResponseBody(context.Background(), req("1", "x"), endOK(), a.GetMetadata())
	f.ResponseBody(context.Background(), req("2", "y"), endOK(), a.GetMetadata())
	complete(t, f, "5", "v", a)
	assert.Equal(t, []string{"z", "w", "v"}, resident(f, a), "the next store trims back to capacity")
}

func TestScrapeCorrection(t *testing.T) {
	f := newFilter(t)
	ctx := context.Background()
	now := time.Now()
	a := endpoint("a", &fwkdl.Metrics{ActiveModels: map[string]int{citations: 1, baseModel: 1}, MaxActiveModels: 2, UpdateTime: now})
	b := endpoint("b", &fwkdl.Metrics{})

	got := f.Filter(ctx, req("r", citations), []fwksched.Endpoint{a, b})
	assert.Equal(t, []string{"a"}, names(got))
	assert.Equal(t, []string{citations}, resident(f, a))
	f.mu.Lock()
	assert.Equal(t, 2, f.slotsOf(f.pods[a.GetMetadata().ID]), "MaxActiveModels replaces fallbackSlots")
	f.mu.Unlock()

	// Absence from ActiveModels does not remove; the same UpdateTime is not re-applied.
	a = endpoint("a", &fwkdl.Metrics{ActiveModels: map[string]int{uncertainty: 1}, UpdateTime: now})
	f.Filter(ctx, req("r", "x"), []fwksched.Endpoint{a})
	assert.Equal(t, []string{citations}, resident(f, a))

	a = endpoint("a", &fwkdl.Metrics{ActiveModels: map[string]int{uncertainty: 1}, UpdateTime: now.Add(time.Second)})
	f.Filter(ctx, req("r", "x"), []fwksched.Endpoint{a})
	assert.Equal(t, []string{citations, uncertainty}, resident(f, a))
}

func TestPinsAndInflightReleased(t *testing.T) {
	f := newFilter(t)
	a := endpoint("a", &fwkdl.Metrics{})
	for i := range 5 {
		complete(t, f, strconv.Itoa(i), "x", a)
	}
	assert.Empty(t, f.inflight)
	assert.Empty(t, f.pods[a.GetMetadata().ID].pins)
}

func TestConcurrentAccess(t *testing.T) {
	f := newFilter(t)
	eps := []fwksched.Endpoint{endpoint("a", &fwkdl.Metrics{}), endpoint("b", &fwkdl.Metrics{})}
	adapters := []string{"x", "y", "z", "w", "v"}
	var wg sync.WaitGroup
	for i := range 200 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.Background()
			id := strconv.Itoa(i)
			r := req(id, adapters[i%len(adapters)])
			picked := f.Filter(ctx, r, eps)
			ep := picked[i%len(picked)]
			_ = f.PreRequest(ctx, r, resultFor(ep))
			f.ResponseHeader(ctx, r, ok200(), ep.GetMetadata())
			f.ResponseBody(ctx, r, endOK(), ep.GetMetadata())
		}(i)
	}
	wg.Wait()
	assert.Empty(t, f.inflight)
	for _, ep := range eps {
		assert.Empty(t, f.pods[ep.GetMetadata().ID].pins)
		assert.Empty(t, f.pods[ep.GetMetadata().ID].loading)
	}
}
