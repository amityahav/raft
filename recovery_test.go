// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"fmt"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLookupRecoverEntry_FileLogStore(t *testing.T) {
	store := testStore(t)
	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	t.Run("have", func(t *testing.T) {
		result, entry, err := lookupRecoverEntry(store, 3, 1)
		require.NoError(t, err)
		assert.Equal(t, RecoveryHave, result)
		require.NotNil(t, entry)
		assert.Equal(t, uint64(3), entry.Index)
		assert.Equal(t, uint64(1), entry.Term)
	})

	t.Run("dontHave missing index", func(t *testing.T) {
		result, entry, err := lookupRecoverEntry(store, 99, 1)
		require.NoError(t, err)
		assert.Equal(t, RecoveryDontHave, result)
		assert.Nil(t, entry)
	})

	t.Run("dontHave wrong term", func(t *testing.T) {
		result, entry, err := lookupRecoverEntry(store, 3, 9)
		require.NoError(t, err)
		assert.Equal(t, RecoveryDontHave, result)
		assert.Nil(t, entry)
	})

	t.Run("haveFaulty", func(t *testing.T) {
		seg := store.findSegment(2)
		require.NotNil(t, seg)
		rec := seg.index[2]
		corruptOff := rec.DataOffset + int64(entryLenSize) + 1
		var b [1]byte
		_, err := seg.file.ReadAt(b[:], corruptOff)
		require.NoError(t, err)
		b[0] ^= 0xFF
		_, err = seg.file.WriteAt(b[:], corruptOff)
		require.NoError(t, err)

		result, entry, err := lookupRecoverEntry(store, 2, 1)
		require.NoError(t, err)
		assert.Equal(t, RecoveryHaveFaulty, result)
		assert.Nil(t, entry)
	})
}

func TestLookupRecoverEntry_InmemStore(t *testing.T) {
	store := NewInmemStore()
	require.NoError(t, store.StoreLogs(testLogs(1, 3)))

	result, entry, err := lookupRecoverEntry(store, 2, 1)
	require.NoError(t, err)
	assert.Equal(t, RecoveryHave, result)
	require.NotNil(t, entry)
	assert.Equal(t, uint64(2), entry.Index)

	result, entry, err = lookupRecoverEntry(store, 2, 7)
	require.NoError(t, err)
	assert.Equal(t, RecoveryDontHave, result)
	assert.Nil(t, entry)

	result, entry, err = lookupRecoverEntry(store, 50, 1)
	require.NoError(t, err)
	assert.Equal(t, RecoveryDontHave, result)
	assert.Nil(t, entry)
}

func TestRecoverEntry_RPC_HaveAndDontHave(t *testing.T) {
	r, trans, addr := startCTRLRaft(t)

	require.Eventually(t, func() bool {
		return r.State() == Leader
	}, 5*time.Second, 10*time.Millisecond)

	fut := r.Apply([]byte("hello"), 2*time.Second)
	require.NoError(t, fut.Error())
	idx := fut.Index()

	var stored Log
	require.NoError(t, r.logs.GetLog(idx, &stored))

	_, peer := NewInmemTransport("")
	peer.Connect(addr, trans)

	hdr := RPCHeader{ProtocolVersion: ProtocolVersionMax}

	var have RecoverEntryResponse
	require.NoError(t, peer.RecoverEntry("peer", addr, &RecoverEntryRequest{
		RPCHeader: hdr,
		Index:     idx,
		Term:      stored.Term,
	}, &have))
	assert.Equal(t, RecoveryHave, have.Result)
	require.NotNil(t, have.Entry)
	assert.Equal(t, idx, have.Entry.Index)
	assert.Equal(t, []byte("hello"), have.Entry.Data)

	var missing RecoverEntryResponse
	require.NoError(t, peer.RecoverEntry("peer", addr, &RecoverEntryRequest{
		RPCHeader: hdr,
		Index:     idx + 100,
		Term:      stored.Term,
	}, &missing))
	assert.Equal(t, RecoveryDontHave, missing.Result)
	assert.Nil(t, missing.Entry)

	var wrongTerm RecoverEntryResponse
	require.NoError(t, peer.RecoverEntry("peer", addr, &RecoverEntryRequest{
		RPCHeader: hdr,
		Index:     idx,
		Term:      stored.Term + 1,
	}, &wrongTerm))
	assert.Equal(t, RecoveryDontHave, wrongTerm.Result)
}

func TestRecoverEntry_RPC_HaveFaulty(t *testing.T) {
	r, trans, addr := startCTRLRaft(t)

	require.Eventually(t, func() bool {
		return r.State() == Leader
	}, 5*time.Second, 10*time.Millisecond)

	fut := r.Apply([]byte("corrupt-me"), 2*time.Second)
	require.NoError(t, fut.Error())
	idx := fut.Index()

	var stored Log
	require.NoError(t, r.logs.GetLog(idx, &stored))

	store, ok := r.logs.(*FileLogStore)
	require.True(t, ok)
	seg := store.findSegment(idx)
	require.NotNil(t, seg)
	rec := seg.index[idx]
	corruptOff := rec.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	_, err := seg.file.ReadAt(b[:], corruptOff)
	require.NoError(t, err)
	b[0] ^= 0xFF
	_, err = seg.file.WriteAt(b[:], corruptOff)
	require.NoError(t, err)

	_, peer := NewInmemTransport("")
	peer.Connect(addr, trans)

	var resp RecoverEntryResponse
	require.NoError(t, peer.RecoverEntry("peer", addr, &RecoverEntryRequest{
		RPCHeader: RPCHeader{ProtocolVersion: ProtocolVersionMax},
		Index:     idx,
		Term:      stored.Term,
	}, &resp))
	assert.Equal(t, RecoveryHaveFaulty, resp.Result)
	assert.Nil(t, resp.Entry)
}

func TestRecoverEntry_RPC_InmemStore(t *testing.T) {
	conf := testCTRLConfig(t)
	addr, trans := NewInmemTransport("")
	logs := NewInmemStore()
	stable := NewInmemStore()
	snaps := NewInmemSnapshotStore()

	cfg := Configuration{Servers: []Server{{
		Suffrage: Voter,
		ID:       conf.LocalID,
		Address:  addr,
	}}}
	require.NoError(t, BootstrapCluster(conf, logs, stable, snaps, trans, cfg))

	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, trans)
	require.NoError(t, err)
	t.Cleanup(func() { _ = r.Shutdown().Error() })

	require.Eventually(t, func() bool {
		return r.State() == Leader
	}, 5*time.Second, 10*time.Millisecond)

	fut := r.Apply([]byte("plain"), 2*time.Second)
	require.NoError(t, fut.Error())
	idx := fut.Index()

	var stored Log
	require.NoError(t, logs.GetLog(idx, &stored))

	_, peer := NewInmemTransport("")
	peer.Connect(addr, trans)

	var resp RecoverEntryResponse
	require.NoError(t, peer.RecoverEntry("peer", addr, &RecoverEntryRequest{
		RPCHeader: RPCHeader{ProtocolVersion: ProtocolVersionMax},
		Index:     idx,
		Term:      stored.Term,
	}, &resp))
	assert.Equal(t, RecoveryHave, resp.Result)
	require.NotNil(t, resp.Entry)
	assert.Equal(t, []byte("plain"), resp.Entry.Data)
}

func TestNetworkTransport_RecoverEntry(t *testing.T) {
	trans1, err := makeTransport(t, false, "localhost:0")
	require.NoError(t, err)
	t.Cleanup(func() { _ = trans1.Close() })
	rpcCh := trans1.Consumer()

	args := RecoverEntryRequest{
		RPCHeader: RPCHeader{ProtocolVersion: ProtocolVersionMax, Addr: []byte("peer")},
		Index:     42,
		Term:      7,
	}
	want := RecoverEntryResponse{
		RPCHeader: RPCHeader{ProtocolVersion: ProtocolVersionMax},
		Result:    RecoveryHave,
		Entry:     &Log{Index: 42, Term: 7, Type: LogCommand, Data: []byte("ok")},
	}

	go func() {
		select {
		case rpc := <-rpcCh:
			req := rpc.Command.(*RecoverEntryRequest)
			if !reflect.DeepEqual(req, &args) {
				t.Errorf("command mismatch: %#v %#v", *req, args)
				return
			}
			rpc.Respond(&want, nil)
		case <-time.After(time.Second):
			t.Errorf("timeout waiting for RecoverEntry")
		}
	}()

	trans2, err := makeTransport(t, false, string(trans1.LocalAddr()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = trans2.Close() })

	var out RecoverEntryResponse
	require.NoError(t, trans2.RecoverEntry("id1", trans1.LocalAddr(), &args, &out))
	assert.Equal(t, want.Result, out.Result)
	require.NotNil(t, out.Entry)
	assert.Equal(t, want.Entry.Index, out.Entry.Index)
	assert.Equal(t, want.Entry.Term, out.Entry.Term)
	assert.Equal(t, want.Entry.Data, out.Entry.Data)
}

func TestInmemTransport_WithRecovery(t *testing.T) {
	var inm interface{} = &InmemTransport{}
	_, ok := inm.(WithRecovery)
	assert.True(t, ok)
	_, ok = inm.(LoopbackTransport)
	assert.True(t, ok)
}

func testCTRLConfig(t *testing.T) *Config {
	t.Helper()
	conf := DefaultConfig()
	conf.LocalID = "ctrl-recovery"
	conf.HeartbeatTimeout = 50 * time.Millisecond
	conf.ElectionTimeout = 50 * time.Millisecond
	conf.LeaderLeaseTimeout = 50 * time.Millisecond
	conf.CommitTimeout = 5 * time.Millisecond
	conf.Logger = newTestLogger(t)
	return conf
}

func startCTRLRaft(t *testing.T) (*Raft, *InmemTransport, ServerAddress) {
	t.Helper()
	dir := t.TempDir()
	conf := testCTRLConfig(t)

	logs, err := NewFileLogStore(filepath.Join(dir, "logs"), FileLogStoreConfig{
		NoSync:               true,
		MaxEntriesPerSegment: 64,
		SegmentSize:          int64(segmentHeaderSize) + 64*int64(identifierSlotSize) + 256*1024,
		Logger:               conf.Logger,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = logs.Close() })

	snaps := NewInmemSnapshotStore()
	stable := NewInmemStore()
	addr, trans := NewInmemTransport("")

	cfg := Configuration{Servers: []Server{{
		Suffrage: Voter,
		ID:       conf.LocalID,
		Address:  addr,
	}}}
	require.NoError(t, BootstrapCluster(conf, logs, stable, snaps, trans, cfg))

	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, trans)
	require.NoError(t, err)
	t.Cleanup(func() { _ = r.Shutdown().Error() })
	return r, trans, addr
}

func TestRecoverDecision(t *testing.T) {
	// 3 voters → quorum 2. One Have is enough to repair (paper §3.4.3 Case 1).
	assert.Equal(t, recoverRepair, recoverDecision(1, 0, 2))
	assert.Equal(t, recoverRepair, recoverDecision(1, 1, 2))
	assert.Equal(t, recoverRepair, recoverDecision(2, 0, 2))
	assert.Equal(t, recoverDiscard, recoverDecision(0, 2, 2))
	assert.Equal(t, recoverAmbiguous, recoverDecision(0, 0, 2))
	assert.Equal(t, recoverAmbiguous, recoverDecision(0, 1, 2))
	// 5 voters → quorum 3. Have still wins even when DontHave is also a majority
	// (paper: first of Case 1 / Case 2; keeping a correct copy is always safe).
	assert.Equal(t, recoverRepair, recoverDecision(1, 3, 3))
	assert.Equal(t, recoverDiscard, recoverDecision(0, 3, 3))
	assert.Equal(t, recoverAmbiguous, recoverDecision(0, 2, 3))
}

func TestFileLogStore_GetLogWithIntegrityRecordsFaulty(t *testing.T) {
	store := testStore(t)
	require.NoError(t, store.StoreLogs(testLogs(1, 3)))
	corruptLogData(t, store, 2)

	var log Log
	status, err := store.GetLogWithIntegrity(2, &log)
	require.NoError(t, err)
	assert.Equal(t, StatusCorrupted, status)

	faulty, err := store.GetFaultyEntries()
	require.NoError(t, err)
	require.Len(t, faulty, 1)
	assert.Equal(t, uint64(2), faulty[0].Index)
}

func TestNewRaft_CorruptedLogDoesNotPanic(t *testing.T) {
	dir := t.TempDir()
	conf := testCTRLConfig(t)
	conf.skipStartup = true

	logs := newTestFileLogStore(t, dir, conf)
	stable := NewInmemStore()
	snaps := newTestChunkedSnaps(t, dir, conf)
	addr, trans := NewInmemTransport("")

	cfg := Configuration{Servers: []Server{{
		Suffrage: Voter, ID: conf.LocalID, Address: addr,
	}}}
	require.NoError(t, BootstrapCluster(conf, logs, stable, snaps, trans, cfg))
	require.NoError(t, logs.StoreLogs([]*Log{{
		Index: 2, Term: 1, Type: LogCommand, Data: []byte("payload"),
	}}))
	corruptLogData(t, logs, 2)

	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, trans)
	require.NoError(t, err)
	t.Cleanup(func() { _ = r.Shutdown().Error() })
	require.NotNil(t, r)
}

func newTestChunkedSnaps(t *testing.T, dir string, conf *Config) *ChunkedFileSnapshotStore {
	t.Helper()
	snaps, err := NewChunkedFileSnapshotStoreWithLogger(filepath.Join(dir, "snaps"), 3, conf.Logger)
	require.NoError(t, err)
	return snaps
}

func newTestFileLogStore(t *testing.T, dir string, conf *Config) *FileLogStore {
	t.Helper()
	logs, err := NewFileLogStore(filepath.Join(dir, "logs"), FileLogStoreConfig{
		NoSync:               true,
		MaxEntriesPerSegment: 64,
		SegmentSize:          int64(segmentHeaderSize) + 64*int64(identifierSlotSize) + 256*1024,
		Logger:               conf.Logger,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = logs.Close() })
	return logs
}

func corruptLogData(t *testing.T, store *FileLogStore, index uint64) {
	t.Helper()
	seg := store.findSegment(index)
	require.NotNil(t, seg)
	rec := seg.index[index]
	off := rec.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	_, err := seg.file.ReadAt(b[:], off)
	require.NoError(t, err)
	b[0] ^= 0xFF
	_, err = seg.file.WriteAt(b[:], off)
	require.NoError(t, err)
}

func connectInmem(t0 *InmemTransport, a0 ServerAddress, t1 *InmemTransport, a1 ServerAddress, t2 *InmemTransport, a2 ServerAddress) {
	t0.Connect(a1, t1)
	t0.Connect(a2, t2)
	t1.Connect(a0, t0)
	t1.Connect(a2, t2)
	t2.Connect(a0, t0)
	t2.Connect(a1, t1)
}

func answerRecover(t *testing.T, trans *InmemTransport, result RecoveryResponse, entry *Log) {
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
			resp := &RecoverEntryResponse{
				RPCHeader: req.RPCHeader,
				Result:    result,
			}
			if result == RecoveryHave && entry != nil {
				copy := *entry
				resp.Entry = &copy
			}
			rpc.Respond(resp, nil)
		}
	}
}

func startRecoverLeader(t *testing.T, orig *Log, peerResult RecoveryResponse, peerEntry *Log) (*Raft, *FileLogStore) {
	t.Helper()
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

	go answerRecover(t, t1, peerResult, peerEntry)
	go answerRecover(t, t2, peerResult, peerEntry)

	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, t0)
	require.NoError(t, err)
	require.True(t, r.ctrlEnabled)
	t.Cleanup(func() { _ = r.Shutdown().Error() })
	return r, logs
}
