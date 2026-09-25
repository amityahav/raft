// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"sync"
	"sync/atomic"
	"time"
)

// recoveryManager drives leader-side runtime log recovery. It is created when a
// node becomes leader (with CTRL enabled) and torn down on step-down. A single
// worker polls GetFaultyEntries and recovers one index at a time. Replication
// and apply never wait on repair: they return an error at a hole and retry
// after the poller has fixed or discarded it.
type recoveryManager struct {
	mu        sync.Mutex
	closed    bool
	stopCh    chan struct{}
	commitCh  chan struct{}
	discardCh chan uint64

	peers  []Server
	quorum int
}

// startRecovery installs a recovery manager and spawns its poller. Must be
// called on the main thread from runLeader after setupLeaderState, and before
// startStopReplication. It is a no-op unless CTRL is enabled.
func (r *Raft) startRecovery() {
	if !r.ctrlEnabled {
		return
	}
	m := &recoveryManager{
		stopCh:    make(chan struct{}),
		commitCh:  r.leaderState.commitCh,
		discardCh: make(chan uint64),
	}
	r.recovery.Store(m)
	r.syncRecoveryMembership()
	r.goFunc(func() { r.runRecovery(m) })
}

// syncRecoveryMembership copies the current voter set and quorum onto the
// recovery manager. Must run on the main thread: r.configurations is not
// safe to read from the worker. Call this after the latest configuration is
// stored and before startStopReplication so new replicators recover against
// the matching membership.
func (r *Raft) syncRecoveryMembership() {
	m := r.recovery.Load()
	if m == nil {
		return
	}
	peers := r.voterPeers()
	quorum := r.quorumSize()
	m.mu.Lock()
	m.peers = peers
	m.quorum = quorum
	m.mu.Unlock()
}

// stopRecovery tears down the recovery manager. Must be called on the main
// thread from the runLeader step-down defer.
func (r *Raft) stopRecovery() {
	m := r.recovery.Swap(nil)
	if m == nil {
		return
	}
	m.mu.Lock()
	m.closed = true
	m.mu.Unlock()
	close(m.stopCh)
}

// runRecovery polls the local faulty set until the manager is torn down or the
// node shuts down. HeartbeatTimeout is the idle interval so recovery is not
// bound to an election.
func (r *Raft) runRecovery(m *recoveryManager) {
	store, ok := r.logs.(CorruptionAwareLogStore)
	if !ok {
		return
	}
	interval := r.config().HeartbeatTimeout
	if interval <= 0 {
		interval = 50 * time.Millisecond
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	r.pollFaultyEntries(m, store)
	for {
		select {
		case <-m.stopCh:
			return
		case <-r.shutdownCh:
			return
		case <-ticker.C:
			r.pollFaultyEntries(m, store)
		}
	}
}

func (r *Raft) pollFaultyEntries(m *recoveryManager, store CorruptionAwareLogStore) {
	m.mu.Lock()
	closed := m.closed
	m.mu.Unlock()
	if closed {
		return
	}

	faulty, err := store.GetFaultyEntries()
	if err != nil {
		r.logger.Error("recovery: list faulty entries", "error", err)
		return
	}
	if len(faulty) == 0 {
		return
	}
	fe := faulty[0]
	r.recoverIndex(m, store, fe.Index, fe.Term)
}

// recoverIndex recovers a single faulty ⟨term, index⟩: re-check the store,
// otherwise query voters and apply FAST'18 §3.4.3: ≥1 Have repairs in place;
// majority DontHave truncates the uncommitted suffix on the main thread;
// otherwise log and leave a TODO (all remaining copies faulty / not enough votes).
func (r *Raft) recoverIndex(m *recoveryManager, store CorruptionAwareLogStore, index, term uint64) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	peers := append([]Server(nil), m.peers...)
	quorum := m.quorum
	m.mu.Unlock()

	var log Log
	status, err := store.GetLogWithIntegrity(index, &log)
	if err != nil {
		r.logger.Error("recovery: read faulty entry", "index", index, "error", err)
		return
	}
	if status == StatusOK && log.Term == term {
		return
	}

	have, dontHave, replica := r.queryVoters(peers, index, term, r.config().ElectionTimeout)
	switch recoverDecision(have, dontHave, quorum) {
	case recoverRepair:
		if replica == nil {
			r.logger.Error("recovery: have response but no copy returned", "index", index)
			return
		}
		if err := store.RepairEntry(replica); err != nil {
			r.logger.Error("recovery: repair failed", "index", index, "error", err)
			return
		}
		r.logger.Info("recovery: repaired faulty log entry", "index", index, "term", term)
		asyncNotifyCh(m.commitCh)

	case recoverDiscard:
		select {
		case m.discardCh <- index:
		case <-m.stopCh:
		}

	default:
		// TODO(ctrl): if every remaining copy is HaveFaulty (or we timed out
		// without a Have or a majority DontHave), the paper stays unavailable
		// until a healthy replica appears. For now just log; the poller will
		// retry on the next tick.
		r.logger.Error("recovery: no intact copy and no majority DontHave; leaving entry unrepaired",
			"index", index, "term", term, "have", have, "dontHave", dontHave, "quorum", quorum)
	}
}

// recoveryDiscardCh is the leaderLoop receive side for uncommitted-suffix
// truncation. A nil channel is ignored by select when CTRL is off.
func (r *Raft) recoveryDiscardCh() <-chan uint64 {
	m := r.recovery.Load()
	if m == nil {
		return nil
	}
	return m.discardCh
}

// handleRecoverDiscard truncates the uncommitted suffix starting at index.
// Must run on the main thread. Mirrors the AppendEntries conflict path:
// DeleteRange through lastLog, roll latest config back to committed if it
// lived in the suffix, then setLastLog. Clamps follower nextIndex so
// replication does not snapshot a missing prev log.
func (r *Raft) handleRecoverDiscard(index uint64) {
	if index <= r.getCommitIndex() {
		r.logger.Error("recovery: refusing to discard at or below commit index",
			"index", index, "commit", r.getCommitIndex())
		return
	}

	lastIdx := r.getLastIndex()
	if err := r.logs.DeleteRange(index, lastIdx); err != nil {
		r.logger.Error("recovery: truncate uncommitted suffix", "from", index, "error", err)
		return
	}
	if r.configurations.latestIndex >= index {
		r.setLatestConfiguration(r.configurations.committed, r.configurations.committedIndex)
	}
	newLast, err := r.logs.LastIndex()
	if err != nil {
		r.logger.Error("recovery: last index after truncate", "error", err)
		return
	}
	if newLast > 0 {
		var lastLog Log
		status, err := readLogEntry(r.logs, newLast, &lastLog)
		if err != nil {
			r.logger.Error("recovery: read new last log after truncate", "index", newLast, "error", err)
			return
		}
		if status != StatusOK {
			r.logger.Error("recovery: new last log is faulty after truncate", "index", newLast)
			return
		}
		r.setLastLog(lastLog.Index, lastLog.Term)
	} else {
		r.setLastLog(0, 0)
	}

	if r.leaderState.inflight != nil {
		for e := r.leaderState.inflight.Front(); e != nil; {
			next := e.Next()
			fut := e.Value.(*logFuture)
			if fut.log.Index >= index {
				fut.respond(ErrLogNotFound)
				r.leaderState.inflight.Remove(e)
			}
			e = next
		}
	}
	nextFloor := newLast + 1
	for _, f := range r.leaderState.replState {
		if atomic.LoadUint64(&f.nextIndex) > nextFloor {
			atomic.StoreUint64(&f.nextIndex, nextFloor)
		}
		asyncNotifyCh(f.triggerCh)
	}

	r.logger.Info("recovery: discarded uncommitted faulty suffix",
		"from", index, "to", lastIdx, "newLast", newLast)
}

// readReplicationLog reads a log entry for replication. When CTRL is enabled and
// the entry is faulty, it returns ErrCorruptedEntry after the store has noted
// it. The leader poller repairs the faulty set; this call does not wait.
func (r *Raft) readReplicationLog(index uint64, out *Log) error {
	if !r.ctrlEnabled {
		return r.logs.GetLog(index, out)
	}
	status, err := readLogEntry(r.logs, index, out)
	if err != nil {
		return err
	}
	if status == StatusOK {
		return nil
	}
	return ErrCorruptedEntry
}
