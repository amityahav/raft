// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// startRuntimeRecovery boots a 3-voter node whose local entry `orig` is
// corrupted, then puts it into leader state with the runtime recovery worker
// running (without going through a real election). The two peers answer
// RecoverEntry with peerResult / peerEntry.
func startRuntimeRecovery(t *testing.T, orig *Log, peerResult RecoveryResponse, peerEntry *Log) (*Raft, *FileLogStore) {
	t.Helper()
	r, logs := startRecoverLeader(t, orig, peerResult, peerEntry)
	r.setState(Leader)
	r.setupLeaderState()
	r.startRecovery()
	t.Cleanup(func() { r.stopRecovery() })
	return r, logs
}

// gatedAnswerRecover answers RecoverEntry only after release is closed, and
// counts how many requests it received. Lets a test observe coalescing while
// the worker is blocked mid-query.
func gatedAnswerRecover(t *testing.T, trans *InmemTransport, result RecoveryResponse, entry *Log, release <-chan struct{}, count *int32) {
	t.Helper()
	done := make(chan struct{})
	t.Cleanup(func() { close(done) })
	for {
		select {
		case <-done:
			return
		case rpc, ok := <-trans.Consumer():
			if !ok {
				return
			}
			req, isReq := rpc.Command.(*RecoverEntryRequest)
			if !isReq {
				rpc.Respond(nil, fmt.Errorf("unexpected command %T", rpc.Command))
				continue
			}
			atomic.AddInt32(count, 1)
			select {
			case <-release:
			case <-done:
				return
			}
			resp := &RecoverEntryResponse{RPCHeader: req.RPCHeader, Result: result}
			if result == RecoveryHave && entry != nil {
				cp := *entry
				resp.Entry = &cp
			}
			rpc.Respond(resp, nil)
		}
	}
}

// TestRuntimeRecovery_RepairViaWorker exercises the on-demand path: reading a
// corrupted entry for replication drives the worker to repair it from a peer
// and returns the healed entry, without blocking on a full election.
func TestRuntimeRecovery_RepairViaWorker(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("hello")}
	r, logs := startRuntimeRecovery(t, orig, RecoveryHave, orig)

	var out Log
	require.NoError(t, r.readReplicationLog(2, &out))
	assert.Equal(t, orig.Data, out.Data)

	// The store itself must now be healed.
	var chk Log
	require.NoError(t, logs.GetLog(2, &chk))
	assert.Equal(t, orig.Data, chk.Data)

	faulty, err := logs.GetFaultyEntries()
	require.NoError(t, err)
	assert.Empty(t, faulty)
}

// TestRuntimeRecovery_CoalescesConcurrentRequests proves that two callers that
// hit the same faulty index share one future and cause exactly one query round.
func TestRuntimeRecovery_CoalescesConcurrentRequests(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("coalesce")}

	dir := t.TempDir()
	conf := testCTRLConfig(t)
	conf.skipStartup = true
	conf.LocalID = "n0"

	logs := newTestFileLogStore(t, dir, conf)
	stable := NewInmemStore()
	snaps := newTestChunkedSnaps(t, dir, conf)

	addr0, t0 := NewInmemTransport("")
	addr1, t1 := NewInmemTransport("")
	addr2, t2 := NewInmemTransport("")
	connectInmem(t0, addr0, t1, addr1, t2, addr2)

	cfg := Configuration{Servers: []Server{
		{Suffrage: Voter, ID: "n0", Address: addr0},
		{Suffrage: Voter, ID: "n1", Address: addr1},
		{Suffrage: Voter, ID: "n2", Address: addr2},
	}}
	require.NoError(t, BootstrapCluster(conf, logs, stable, snaps, t0, cfg))
	require.NoError(t, logs.StoreLogs([]*Log{orig}))
	corruptLogData(t, logs, orig.Index)

	release := make(chan struct{})
	var n1count, n2count int32
	go gatedAnswerRecover(t, t1, RecoveryHave, orig, release, &n1count)
	go gatedAnswerRecover(t, t2, RecoveryHave, orig, release, &n2count)

	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, t0)
	require.NoError(t, err)
	require.True(t, r.ctrlEnabled)
	t.Cleanup(func() { _ = r.Shutdown().Error() })

	r.setState(Leader)
	r.setupLeaderState()
	r.startRecovery()
	t.Cleanup(func() { r.stopRecovery() })

	// First request enqueues work; the worker will block inside queryVoters
	// against the gated peers. A second request for the same index gets its own
	// future but must NOT start a new query round.
	f1 := r.requestRecovery(2, 1)
	require.NotNil(t, f1)

	// Wait until at least one peer has received the query, so the worker is
	// mid-flight and inflight[2] is guaranteed present.
	require.Eventually(t, func() bool {
		return atomic.LoadInt32(&n1count)+atomic.LoadInt32(&n2count) >= 1
	}, time.Second, 5*time.Millisecond)

	f2 := r.requestRecovery(2, 1)
	require.NotNil(t, f2)

	close(release)
	require.NoError(t, f1.Error())
	require.NoError(t, f2.Error())
	assert.Equal(t, orig.Data, f1.log.Data)
	assert.Equal(t, orig.Data, f2.log.Data)

	assert.Equal(t, int32(1), atomic.LoadInt32(&n1count), "peer n1 must be queried once despite two waiters")
	assert.Equal(t, int32(1), atomic.LoadInt32(&n2count), "peer n2 must be queried once despite two waiters")
}

// TestRuntimeRecovery_StepsDownOnUncommitted verifies that an uncommitted
// faulty entry (majority DontHave) makes the runtime worker step the leader
// down instead of truncating on a background goroutine.
func TestRuntimeRecovery_StepsDownOnUncommitted(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("uncommitted")}
	r, _ := startRuntimeRecovery(t, orig, RecoveryDontHave, nil)

	f := r.requestRecovery(2, 1)
	require.NotNil(t, f)
	assert.ErrorIs(t, f.Error(), ErrRecoveryAmbiguous)

	select {
	case <-r.leaderState.stepDown:
	case <-time.After(time.Second):
		t.Fatal("expected step-down signal")
	}
}

// TestRuntimeRecovery_ReadFallsBackWhenNoManager ensures readReplicationLog
// returns the corruption error (rather than blocking forever) when there is no
// active recovery manager, e.g. a corrupted read outside of leadership.
func TestRuntimeRecovery_ReadFallsBackWhenNoManager(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("no-manager")}
	r, _ := startRecoverLeader(t, orig, RecoveryHave, orig)
	// Do not start a recovery manager.

	var out Log
	err := r.readReplicationLog(2, &out)
	assert.ErrorIs(t, err, ErrCorruptedEntry)
}
