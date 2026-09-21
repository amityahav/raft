// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"fmt"
	"sync"
)

// recoverFuture is a caller's handle for the recovery of one faulty log index.
// It follows the codebase future pattern (embedded deferError with respond /
// Error), and additionally carries the repaired entry. Many callers that hit
// the same index each get their own future; the worker responds to all of them
// after a single RecoverEntry round.
type recoverFuture struct {
	deferError
	index uint64
	term  uint64
	log   *Log
}

// recoveryManager drives leader-side runtime log recovery. It is created when a
// node becomes leader (with CTRL enabled) and torn down on step-down. A single
// worker goroutine consumes indices from workCh, so faulty entries are recovered
// one at a time; producers (replication and apply paths) coalesce on inflight so
// multiple followers that hit the same hole issue only one RecoverEntry round.
type recoveryManager struct {
	mu       sync.Mutex
	inflight map[uint64][]*recoverFuture
	closed   bool

	workCh    chan FaultyEntry
	stopCh    chan struct{}
	commitCh  chan struct{}
	discardCh chan uint64

	peers  []Server
	quorum int
}

// startRecovery installs a recovery manager and spawns its worker. Must be
// called on the main thread from runLeader after setupLeaderState, and before
// startStopReplication, so replicators never observe a nil manager or a stale
// membership snapshot. It is a no-op unless CTRL is enabled.
func (r *Raft) startRecovery() {
	if !r.ctrlEnabled {
		return
	}
	m := &recoveryManager{
		inflight: make(map[uint64][]*recoverFuture),
		workCh:   make(chan FaultyEntry, 64),
		stopCh:   make(chan struct{}),
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

// stopRecovery tears down the recovery manager and fails any inflight futures so
// no producer blocks forever. Must be called on the main thread from the
// runLeader step-down defer.
func (r *Raft) stopRecovery() {
	m := r.recovery.Swap(nil)
	if m == nil {
		return
	}
	m.mu.Lock()
	m.closed = true
	for idx, waiters := range m.inflight {
		for _, f := range waiters {
			f.respond(ErrLeadershipLost)
		}
		delete(m.inflight, idx)
	}
	m.mu.Unlock()
	close(m.stopCh)
}

// newRecoverFuture builds a future wired to the shutdown channel so Error()
// unblocks on shutdown, matching the codebase future pattern.
func (r *Raft) newRecoverFuture(index, term uint64) *recoverFuture {
	f := &recoverFuture{index: index, term: term}
	f.init()
	f.ShutdownCh = r.shutdownCh
	return f
}

// requestRecovery returns a future for the given faulty index. Each caller gets
// its own future, but callers for the same index coalesce onto a single
// RecoverEntry round (only the first waiter enqueues work). Returns nil if there
// is no active recovery manager.
func (r *Raft) requestRecovery(index, term uint64) *recoverFuture {
	m := r.recovery.Load()
	if m == nil {
		return nil
	}

	f := r.newRecoverFuture(index, term)

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		f.respond(ErrLeadershipLost)
		return f
	}
	first := len(m.inflight[index]) == 0
	m.inflight[index] = append(m.inflight[index], f)
	m.mu.Unlock()

	if !first {
		// A round is already in flight (or queued) for this index.
		return f
	}
	// The waiter is registered, so teardown (which drains inflight) will
	// respond to it even if the worker never sees this send.
	select {
	case m.workCh <- FaultyEntry{Index: index, Term: term}:
	case <-m.stopCh:
	}
	return f
}

// completeRecovery responds to every waiter for an index if it is still
// inflight. It is a no-op if teardown already drained it, so the worker and
// teardown can never respond twice to the same future.
func (r *Raft) completeRecovery(m *recoveryManager, index uint64, log *Log, err error) {
	m.mu.Lock()
	waiters, ok := m.inflight[index]
	if ok {
		delete(m.inflight, index)
	}
	m.mu.Unlock()
	for _, f := range waiters {
		f.log = log
		f.respond(err)
	}
}

// runRecovery is the worker goroutine. It processes one faulty index at a time
// until the manager is torn down or the node shuts down.
func (r *Raft) runRecovery(m *recoveryManager) {
	store, ok := r.logs.(CorruptionAwareLogStore)
	if !ok {
		return
	}
	for {
		select {
		case <-m.stopCh:
			return
		case <-r.shutdownCh:
			return
		case fe := <-m.workCh:
			r.recoverIndex(m, store, fe.Index, fe.Term)
		}
	}
}

// recoverIndex recovers a single faulty ⟨term, index⟩: re-check the store
// (another index's recovery may already have fixed it), otherwise query voters
// and apply FAST'18 §3.4.3: ≥1 Have repairs in place; majority DontHave
// truncates this uncommitted suffix; otherwise log and leave a TODO (all
// remaining copies faulty / not enough votes).
func (r *Raft) recoverIndex(m *recoveryManager, store CorruptionAwareLogStore, index, term uint64) {
	m.mu.Lock()
	_, ok := m.inflight[index]
	peers := append([]Server(nil), m.peers...)
	quorum := m.quorum
	m.mu.Unlock()
	if !ok {
		return // already completed or torn down
	}

	// Re-check: replication to another follower may have already recovered
	// this exact index, or a truncation may have removed it entirely.
	var log Log
	status, err := store.GetLogWithIntegrity(index, &log)
	if err != nil {
		r.completeRecovery(m, index, nil, err)
		return
	}
	if status == StatusOK && log.Term == term {
		cp := log
		r.completeRecovery(m, index, &cp, nil)
		return
	}

	have, dontHave, replica := r.queryVoters(peers, index, term, r.config().ElectionTimeout)
	switch recoverDecision(have, dontHave, quorum) {
	case recoverRepair:
		if replica == nil {
			r.completeRecovery(m, index, nil, fmt.Errorf("have response for entry %d but no copy returned", index))
			return
		}
		if err := store.RepairEntry(replica); err != nil {
			r.completeRecovery(m, index, nil, fmt.Errorf("repair entry %d: %w", index, err))
			return
		}
		r.logger.Info("recovery: repaired faulty log entry", "index", index, "term", term)
		cp := *replica
		r.completeRecovery(m, index, &cp, nil)
		asyncNotifyCh(m.commitCh)

	case recoverDiscard:
		// Truncation and config rollback belong on the main thread
		// (same as AppendEntries suffix conflict). Waiters are completed
		// there after the suffix is gone.
		select {
		case m.discardCh <- index:
		case <-m.stopCh:
			r.completeRecovery(m, index, nil, ErrLeadershipLost)
		}

	default:
		// TODO(ctrl): if every remaining copy is HaveFaulty (or we timed out
		// without a Have or a majority DontHave), the paper stays unavailable
		// until a healthy replica appears. For now just log and fail waiters
		// so replication does not block forever.
		r.logger.Error("recovery: no intact copy and no majority DontHave; leaving entry unrepaired",
			"index", index, "term", term, "have", have, "dontHave", dontHave, "quorum", quorum)
		r.completeRecovery(m, index, nil, ErrRecoveryAmbiguous)
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
// lived in the suffix, then setLastLog. Also fails inflight Apply futures
// and recovery waiters for that index.
func (r *Raft) handleRecoverDiscard(index uint64) {
	m := r.recovery.Load()
	if m == nil {
		return
	}

	if index <= r.getCommitIndex() {
		err := fmt.Errorf("majority dont-have for index %d which is at or below commit index %d",
			index, r.getCommitIndex())
		r.completeRecovery(m, index, nil, err)
		return
	}

	lastIdx := r.getLastIndex()
	if err := r.logs.DeleteRange(index, lastIdx); err != nil {
		r.completeRecovery(m, index, nil, fmt.Errorf("truncate uncommitted suffix from %d: %w", index, err))
		return
	}
	if r.configurations.latestIndex >= index {
		r.setLatestConfiguration(r.configurations.committed, r.configurations.committedIndex)
	}
	newLast, err := r.logs.LastIndex()
	if err != nil {
		r.completeRecovery(m, index, nil, err)
		return
	}
	if newLast > 0 {
		var lastLog Log
		status, err := readLogEntry(r.logs, newLast, &lastLog)
		if err != nil {
			r.completeRecovery(m, index, nil, fmt.Errorf("read new last log %d after truncate: %w", newLast, err))
			return
		}
		if status != StatusOK {
			r.completeRecovery(m, index, nil, fmt.Errorf("new last log %d is faulty after truncate", newLast))
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
	for _, f := range r.leaderState.replState {
		asyncNotifyCh(f.triggerCh)
	}

	r.logger.Info("recovery: discarded uncommitted faulty suffix",
		"from", index, "to", lastIdx, "newLast", newLast)
	r.completeRecovery(m, index, nil, ErrLogNotFound)
}

// readReplicationLog reads a log entry for replication. When CTRL is enabled and
// the entry is corrupted, it drives on-demand recovery via the leader worker and
// blocks the calling replication goroutine (only) until the entry is repaired or
// recovery fails. Heartbeats run on a separate goroutine and are unaffected.
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

	// Corrupted: out.Term/out.Index were populated from the identifier.
	f := r.requestRecovery(index, out.Term)
	if f == nil {
		return ErrCorruptedEntry
	}
	if err := f.Error(); err != nil {
		return err
	}
	*out = *f.log
	return nil
}
