// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func startCTRLNode(t *testing.T, extra []*Log) (*Raft, *FileLogStore) {
	t.Helper()
	dir := t.TempDir()
	conf := testCTRLConfig(t)
	conf.skipStartup = true
	conf.LocalID = "n0"

	logs := newTestFileLogStore(t, dir, conf)
	stable := NewInmemStore()
	snaps := newTestChunkedSnaps(t, dir, conf)
	addr, trans := NewInmemTransport("")

	cfg := Configuration{Servers: []Server{
		{Suffrage: Voter, ID: "n0", Address: addr},
	}}
	require.NoError(t, BootstrapCluster(conf, logs, stable, snaps, trans, cfg))
	if len(extra) > 0 {
		require.NoError(t, logs.StoreLogs(extra))
	}

	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, trans)
	require.NoError(t, err)
	require.True(t, r.ctrlEnabled)
	t.Cleanup(func() { _ = r.Shutdown().Error() })
	return r, logs
}

func callAppendEntries(r *Raft, req *AppendEntriesRequest) *AppendEntriesResponse {
	respCh := make(chan RPCResponse, 1)
	r.appendEntries(RPC{Command: req, RespChan: respCh}, req)
	out := <-respCh
	return out.Response.(*AppendEntriesResponse)
}

func testRepl(term, nextIndex uint64) *followerReplication {
	return &followerReplication{
		currentTerm: term,
		nextIndex:   nextIndex,
		triggerCh:   make(chan struct{}, 1),
		notify:      make(map[*verifyFuture]struct{}),
	}
}

func TestFollowerRecovery_ReportsInteriorHoleOnHeartbeat(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("hole")}
	r, logs := startCTRLNode(t, []*Log{orig})
	corruptLogData(t, logs, orig.Index)
	var probe Log
	assert.ErrorIs(t, logs.GetLog(2, &probe), ErrCorruptedEntry)

	resp := callAppendEntries(r, &AppendEntriesRequest{
		RPCHeader: r.getRPCHeader(),
		Term:      1,
	})
	require.True(t, resp.Success)
	require.Len(t, resp.FaultyEntries, 1)
	assert.Equal(t, uint64(2), resp.FaultyEntries[0].Index)
	assert.Equal(t, uint64(1), resp.FaultyEntries[0].Term)
}

func TestFollowerRecovery_RepairThenProcessLogs(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("apply-me")}
	r, logs := startCTRLNode(t, []*Log{orig})
	corruptLogData(t, logs, orig.Index)
	var probe Log
	require.ErrorIs(t, logs.GetLog(2, &probe), ErrCorruptedEntry)

	r.setLastApplied(1)
	resp := callAppendEntries(r, &AppendEntriesRequest{
		RPCHeader:         r.getRPCHeader(),
		Term:              1,
		RepairEntries:     []*Log{orig},
		LeaderCommitIndex: 2,
	})
	require.True(t, resp.Success)
	assert.Empty(t, resp.FaultyEntries)

	var got Log
	require.NoError(t, logs.GetLog(2, &got))
	assert.Equal(t, orig.Data, got.Data)
	assert.Equal(t, uint64(2), r.getLastApplied())
}

func TestFollowerRecovery_UnknownTermDiscardsSuffix(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("old")}
	later := &Log{Index: 3, Term: 1, Type: LogCommand, Data: []byte("tail")}
	r, logs := startCTRLNode(t, []*Log{orig, later})
	corruptLogData(t, logs, orig.Index)
	var probe Log
	require.ErrorIs(t, logs.GetLog(2, &probe), ErrCorruptedEntry)

	resp := callAppendEntries(r, &AppendEntriesRequest{
		RPCHeader:   r.getRPCHeader(),
		Term:        1,
		DiscardFrom: 2,
	})
	require.True(t, resp.Success)

	assert.ErrorIs(t, logs.GetLog(2, &probe), ErrLogNotFound)
	assert.ErrorIs(t, logs.GetLog(3, &probe), ErrLogNotFound)
	last, err := logs.LastIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(1), last)
}

func TestFollowerRecovery_RepairOtherTermDiscardsNotOverwrites(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("old")}
	r, logs := startCTRLNode(t, []*Log{orig})
	corruptLogData(t, logs, orig.Index)
	var probe Log
	require.ErrorIs(t, logs.GetLog(2, &probe), ErrCorruptedEntry)

	resp := callAppendEntries(r, &AppendEntriesRequest{
		RPCHeader: r.getRPCHeader(),
		Term:      1,
		RepairEntries: []*Log{{
			Index: 2, Term: 2, Type: LogCommand, Data: []byte("new-term"),
		}},
	})
	require.True(t, resp.Success)
	assert.ErrorIs(t, logs.GetLog(2, &probe), ErrLogNotFound,
		"stale ⟨term, index⟩ must be discarded, not repaired under a new term")
}

func TestFollowerRecovery_PrevLogCorruptRejectsAndReports(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("prev")}
	r, logs := startCTRLNode(t, []*Log{orig})
	corruptLogData(t, logs, orig.Index)

	resp := callAppendEntries(r, &AppendEntriesRequest{
		RPCHeader:    r.getRPCHeader(),
		Term:         1,
		PrevLogEntry: 2,
		PrevLogTerm:  1,
		Entries: []*Log{{
			Index: 3, Term: 1, Type: LogCommand, Data: []byte("next"),
		}},
	})
	assert.False(t, resp.Success)
	assert.True(t, resp.NoRetryBackoff)
	require.Len(t, resp.FaultyEntries, 1)
	assert.Equal(t, uint64(2), resp.FaultyEntries[0].Index)

	var got Log
	assert.ErrorIs(t, logs.GetLog(3, &got), ErrLogNotFound, "must not append past a corrupt prev")
}

func TestFollowerRecovery_RepairPrevThenAppend(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("prev")}
	r, logs := startCTRLNode(t, []*Log{orig})
	corruptLogData(t, logs, orig.Index)

	next := &Log{Index: 3, Term: 1, Type: LogCommand, Data: []byte("next")}
	resp := callAppendEntries(r, &AppendEntriesRequest{
		RPCHeader:     r.getRPCHeader(),
		Term:          1,
		PrevLogEntry:  2,
		PrevLogTerm:   1,
		RepairEntries: []*Log{orig},
		Entries:       []*Log{next},
	})
	require.True(t, resp.Success)
	var got Log
	require.NoError(t, logs.GetLog(2, &got))
	assert.Equal(t, orig.Data, got.Data)
	require.NoError(t, logs.GetLog(3, &got))
	assert.Equal(t, next.Data, got.Data)
}

func TestFollowerRecovery_LeaderSendsRepairOnSameTerm(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("copy")}
	r, _ := startCTRLNode(t, []*Log{orig})
	r.setupLeaderState()
	s := testRepl(1, 3)

	r.handleFollowerFaults(s, &AppendEntriesResponse{
		Term: 1,
		FaultyEntries: []FaultyEntry{
			{Index: 2, Term: 1, Status: StatusCorrupted},
		},
	})
	assert.True(t, s.takeRepairKick())

	var req AppendEntriesRequest
	require.NoError(t, r.setupAppendEntries(s, &req, atomic.LoadUint64(&s.nextIndex), r.getLastIndex()))
	require.Len(t, req.RepairEntries, 1)
	assert.Equal(t, orig.Data, req.RepairEntries[0].Data)
	assert.Zero(t, req.DiscardFrom)
}

func TestFollowerRecovery_LeaderDiscardsUnknownIndex(t *testing.T) {
	r, _ := startCTRLNode(t, nil)
	r.setupLeaderState()
	s := testRepl(1, 2)

	r.handleFollowerFaults(s, &AppendEntriesResponse{
		Term: 1,
		FaultyEntries: []FaultyEntry{
			{Index: 9, Term: 1, Status: StatusCorrupted},
		},
	})
	assert.True(t, s.takeRepairKick())

	var req AppendEntriesRequest
	require.NoError(t, r.setupAppendEntries(s, &req, atomic.LoadUint64(&s.nextIndex), r.getLastIndex()))
	assert.Equal(t, uint64(9), req.DiscardFrom)
	assert.Empty(t, req.RepairEntries)
}

func TestFollowerRecovery_LeaderOtherTermLowersNextIndex(t *testing.T) {
	orig := &Log{Index: 2, Term: 2, Type: LogCommand, Data: []byte("new")}
	r, _ := startCTRLNode(t, []*Log{orig})
	r.setupLeaderState()
	s := testRepl(2, 10)

	r.handleFollowerFaults(s, &AppendEntriesResponse{
		Term: 2,
		FaultyEntries: []FaultyEntry{
			{Index: 2, Term: 1, Status: StatusCorrupted},
		},
	})
	assert.Equal(t, uint64(2), atomic.LoadUint64(&s.nextIndex),
		"other term at the same index uses the Entries conflict path")
	assert.True(t, s.takeRepairKick())

	var req AppendEntriesRequest
	require.NoError(t, r.setupAppendEntries(s, &req, atomic.LoadUint64(&s.nextIndex), r.getLastIndex()))
	assert.Empty(t, req.RepairEntries)
	assert.Zero(t, req.DiscardFrom)
}

func TestFollowerRecovery_IgnoresStaleResponseTerm(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("copy")}
	r, _ := startCTRLNode(t, []*Log{orig})
	r.setupLeaderState()
	s := testRepl(3, 3)

	older := &AppendEntriesResponse{
		Term:          2,
		FaultyEntries: []FaultyEntry{{Index: 2, Term: 1}},
	}
	r.handleFollowerFaults(s, older)
	assert.False(t, s.takeRepairKick())

	newer := &AppendEntriesResponse{
		Term:          4,
		FaultyEntries: []FaultyEntry{{Index: 2, Term: 1}},
	}
	r.handleFollowerFaults(s, newer)
	assert.False(t, s.takeRepairKick())

	var req AppendEntriesRequest
	require.NoError(t, r.setupAppendEntries(s, &req, atomic.LoadUint64(&s.nextIndex), r.getLastIndex()))
	assert.Empty(t, req.RepairEntries)
}

func TestFollowerRecovery_PipelineConsumesFaultyEntries(t *testing.T) {
	orig := &Log{Index: 2, Term: 1, Type: LogCommand, Data: []byte("copy")}
	r, _ := startCTRLNode(t, []*Log{orig})
	r.setupLeaderState()
	s := testRepl(1, 3)

	resp := &AppendEntriesResponse{
		Term:          1,
		Success:       true,
		FaultyEntries: []FaultyEntry{{Index: 2, Term: 1, Status: StatusCorrupted}},
	}
	r.handleFollowerFaults(s, resp)
	require.True(t, s.takeRepairKick(), "pipelineDecode must abort so replicateTo can send RepairEntries")

	var req AppendEntriesRequest
	require.NoError(t, r.setupAppendEntries(s, &req, atomic.LoadUint64(&s.nextIndex), r.getLastIndex()))
	require.Len(t, req.RepairEntries, 1)
}

func TestFollowerRecovery_CTRLOffLeavesFieldsEmpty(t *testing.T) {
	conf := inmemConfig(t)
	conf.skipStartup = true
	conf.LocalID = "n0"
	logs := NewInmemStore()
	stable := NewInmemStore()
	snaps := NewInmemSnapshotStore()
	addr, trans := NewInmemTransport("")
	cfg := Configuration{Servers: []Server{
		{Suffrage: Voter, ID: "n0", Address: addr},
	}}
	require.NoError(t, BootstrapCluster(conf, logs, stable, snaps, trans, cfg))
	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, trans)
	require.NoError(t, err)
	t.Cleanup(func() { _ = r.Shutdown().Error() })
	require.False(t, r.ctrlEnabled)

	resp := callAppendEntries(r, &AppendEntriesRequest{
		RPCHeader: r.getRPCHeader(),
		Term:      r.getCurrentTerm(),
	})
	assert.Empty(t, resp.FaultyEntries)
}
