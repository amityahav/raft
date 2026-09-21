// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"container/list"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// startRuntimeRecovery boots a 3-voter node whose local entry `orig` is
// corrupted, then puts it into leader state with the runtime recovery poller
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

func waitUntilRepaired(t *testing.T, logs *FileLogStore, index uint64) {
	t.Helper()
	require.Eventually(t, func() bool {
		var got Log
		return logs.GetLog(index, &got) == nil
	}, 2*time.Second, 5*time.Millisecond)
}

// TestRuntimeRecovery_PollerRepairsFaultyEntry is the production path: a
// corrupted local copy is listed in GetFaultyEntries, the poller repairs it
// from a peer, and replication can then read the healed entry. The hot path
// itself never waits.
func TestRuntimeRecovery_PollerRepairsFaultyEntry(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("hello")}
	r, logs := startRuntimeRecovery(t, orig, RecoveryHave, orig)

	var out Log
	err := r.readReplicationLog(2, &out)
	if err != nil {
		assert.ErrorIs(t, err, ErrCorruptedEntry)
	}

	waitUntilRepaired(t, logs, 2)

	require.NoError(t, r.readReplicationLog(2, &out))
	assert.Equal(t, orig.Data, out.Data)

	var chk Log
	require.NoError(t, logs.GetLog(2, &chk))
	assert.Equal(t, orig.Data, chk.Data)

	faulty, err := logs.GetFaultyEntries()
	require.NoError(t, err)
	assert.Empty(t, faulty)
}

// TestRuntimeRecovery_PollerIssuesOneRound proves the poller recovers a
// single index with one RecoverEntry query to each voter, then goes idle.
func TestRuntimeRecovery_PollerIssuesOneRound(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("one-round")}

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
	close(release)
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

	waitUntilRepaired(t, logs, 2)

	assert.Equal(t, int32(1), atomic.LoadInt32(&n1count), "peer n1 must be queried once")
	assert.Equal(t, int32(1), atomic.LoadInt32(&n2count), "peer n2 must be queried once")
}

// gatedAnswerRecover answers RecoverEntry only after release is closed, and
// counts how many requests it received.
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

// TestRuntimeRecovery_DiscardUncommitted verifies that majority DontHave
// truncates the uncommitted faulty suffix (paper §3.4.3 Case 2) and clamps
// follower nextIndex so the discarded suffix is not treated as a snapshot hole.
func TestRuntimeRecovery_DiscardUncommitted(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("uncommitted")}
	r, logs := startRuntimeRecovery(t, orig, RecoveryDontHave, nil)

	repl := &followerReplication{
		nextIndex: 10,
		triggerCh: make(chan struct{}, 1),
	}
	r.leaderState.replState["n1"] = repl

	m := r.recovery.Load()
	require.NotNil(t, m)
	select {
	case idx := <-m.discardCh:
		assert.Equal(t, uint64(2), idx)
		r.handleRecoverDiscard(idx)
	case <-time.After(2 * time.Second):
		t.Fatal("expected discard to be dispatched to the main thread")
	}

	var got Log
	assert.ErrorIs(t, logs.GetLog(2, &got), ErrLogNotFound)
	last, err := logs.LastIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(1), last)

	assert.Equal(t, uint64(2), atomic.LoadUint64(&repl.nextIndex),
		"nextIndex must be clamped to newLast+1 so we do not snapshot a missing prev")
	select {
	case <-repl.triggerCh:
	default:
		t.Fatal("expected triggerCh after discard")
	}
}

// TestRuntimeRecovery_AllCopiesFaulty is the paper's "remain unavailable"
// case: every queried voter returns HaveFaulty. The poller logs and retries;
// it must not truncate a possibly committed entry.
func TestRuntimeRecovery_AllCopiesFaulty(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("all-faulty")}
	r, logs := startRuntimeRecovery(t, orig, RecoveryHaveFaulty, nil)

	m := r.recovery.Load()
	require.NotNil(t, m)

	time.Sleep(150 * time.Millisecond)
	select {
	case idx := <-m.discardCh:
		t.Fatalf("must not discard when every copy is HaveFaulty, got index %d", idx)
	default:
	}

	var got Log
	status, err := logs.GetLogWithIntegrity(2, &got)
	require.NoError(t, err)
	assert.Equal(t, StatusCorrupted, status)
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

func TestRuntimeRecovery_MembershipTracksConfig(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("cfg")}
	r, _ := startRuntimeRecovery(t, orig, RecoveryHave, orig)

	m := r.recovery.Load()
	require.NotNil(t, m)
	m.mu.Lock()
	assert.Len(t, m.peers, 2)
	assert.Equal(t, 2, m.quorum)
	m.mu.Unlock()

	r.configurations.latest.Servers = append(r.configurations.latest.Servers,
		Server{Suffrage: Voter, ID: "n3", Address: "n3"})
	r.syncRecoveryMembership()

	m.mu.Lock()
	defer m.mu.Unlock()
	assert.Len(t, m.peers, 3, "new voter must be queried on the next recovery round")
	assert.Equal(t, 3, m.quorum, "4 voters → quorum 3")
}

// TestRuntimeRecovery_ApplyResumesAfterRepair: processLogs stops at a CTRL
// hole without panicking; after the poller repairs and nudges commitCh,
// apply continues from lastApplied.
func TestRuntimeRecovery_ApplyResumesAfterRepair(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("apply-me")}
	r, logs := startRecoverLeader(t, orig, RecoveryHave, orig)

	r.setLastApplied(1)
	applied := r.processLogs(2, nil)
	assert.Equal(t, uint64(1), applied)
	assert.Equal(t, uint64(1), r.getLastApplied())

	r.setState(Leader)
	r.setupLeaderState()
	r.startRecovery()
	t.Cleanup(func() { r.stopRecovery() })

	select {
	case <-r.leaderState.commitCh:
	case <-time.After(2 * time.Second):
		t.Fatal("expected commitCh nudge after poller repair")
	}
	waitUntilRepaired(t, logs, 2)

	applied = r.processLogs(2, nil)
	assert.Equal(t, uint64(2), applied)
	assert.Equal(t, uint64(2), r.getLastApplied())
}

// TestRuntimeRecovery_InflightNotRemovedPastHole: leaderLoop must keep
// inflight futures whose index is after the apply hole, and must not respond
// them, so a later commitCh can retry.
func TestRuntimeRecovery_InflightNotRemovedPastHole(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("hole")}
	r, _ := startRecoverLeader(t, orig, RecoveryHave, orig)

	later := &Log{Index: 3, Term: 1, Type: LogCommand, Data: []byte("later")}
	require.NoError(t, r.logs.StoreLogs([]*Log{later}))

	r.setLastApplied(1)
	r.setupLeaderState()

	fut := &logFuture{log: *later}
	fut.init()
	elem := r.leaderState.inflight.PushBack(fut)

	groupReady := []*list.Element{elem}
	groupFutures := map[uint64]*logFuture{3: fut}

	appliedThrough := r.processLogs(3, groupFutures)
	assert.Equal(t, uint64(1), appliedThrough)

	for _, e := range groupReady {
		if e.Value.(*logFuture).log.Index <= appliedThrough {
			r.leaderState.inflight.Remove(e)
		}
	}

	require.Equal(t, 1, r.leaderState.inflight.Len(), "inflight past the hole must stay queued")
	select {
	case err := <-fut.errCh:
		t.Fatalf("must not respond inflight at/after the hole, got %v", err)
	default:
	}
}
