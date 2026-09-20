// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"errors"
	"fmt"
	"time"

	"github.com/hashicorp/go-metrics"
)

const recoverMaxAttempts = 3

var (
	// ErrRecoveryAmbiguous is returned when peer responses do not yield a
	// majority Have or DontHave, so the leader cannot decide whether a
	// faulty entry is committed.
	ErrRecoveryAmbiguous = errors.New("unable to determine whether faulty log entry is committed")
)

// recoverOutcome is the majority decision for one faulty entry.
type recoverOutcome int

const (
	recoverRepair recoverOutcome = iota
	recoverDiscard
	recoverAmbiguous
)

// recoverDecision applies the CTRL quorum rule among voters.
// HaveFaulty (including this node's own copy) counts for neither side.
func recoverDecision(have, dontHave, quorum int) recoverOutcome {
	if have >= quorum {
		return recoverRepair
	}
	if dontHave >= quorum {
		return recoverDiscard
	}
	return recoverAmbiguous
}

// recoverEntry handles an incoming RecoverEntry RPC. It is read-only and
// safe to run in any raft state: it reports whether this node has a healthy
// copy of the requested ⟨term, index⟩.
func (r *Raft) recoverEntry(rpc RPC, req *RecoverEntryRequest) {
	defer metrics.MeasureSince([]string{"raft", "rpc", "recoverEntry"}, time.Now())

	resp := &RecoverEntryResponse{
		RPCHeader: r.getRPCHeader(),
		Result:    RecoveryDontHave,
	}
	var rpcErr error
	defer func() {
		rpc.Respond(resp, rpcErr)
	}()

	result, entry, err := lookupRecoverEntry(r.logs, req.Index, req.Term)
	if err != nil {
		rpcErr = err
		return
	}
	resp.Result = result
	resp.Entry = entry
}

// lookupRecoverEntry classifies a log entry for the CTRL recovery protocol.
//
//	have index, term matches, CRC ok  → Have (+ Entry)
//	have index, term matches, CRC bad → HaveFaulty
//	no index, or term differs         → DontHave
//
// Stores that are not corruption-aware fall back to GetLog, treating a
// successful read as Have.
func lookupRecoverEntry(logs LogStore, index, term uint64) (RecoveryResponse, *Log, error) {
	if store, ok := logs.(CorruptionAwareLogStore); ok {
		var log Log
		status, err := store.GetLogWithIntegrity(index, &log)
		if err != nil {
			if errors.Is(err, ErrLogNotFound) {
				return RecoveryDontHave, nil, nil
			}
			return RecoveryDontHave, nil, fmt.Errorf("read log %d for recovery: %w", index, err)
		}
		if log.Term != term {
			return RecoveryDontHave, nil, nil
		}
		if status == StatusOK {
			return RecoveryHave, &log, nil
		}
		return RecoveryHaveFaulty, nil, nil
	}

	var log Log
	if err := logs.GetLog(index, &log); err != nil {
		if errors.Is(err, ErrLogNotFound) {
			return RecoveryDontHave, nil, nil
		}
		return RecoveryDontHave, nil, fmt.Errorf("read log %d for recovery: %w", index, err)
	}
	if log.Term != term {
		return RecoveryDontHave, nil, nil
	}
	return RecoveryHave, &log, nil
}

// readLogEntry reads a log entry, using integrity APIs when available so a
// corrupted entry returns StatusCorrupted instead of an error.
func readLogEntry(logs LogStore, index uint64, log *Log) (IntegrityStatus, error) {
	if store, ok := logs.(CorruptionAwareLogStore); ok {
		return store.GetLogWithIntegrity(index, log)
	}
	if err := logs.GetLog(index, log); err != nil {
		return StatusOK, err
	}
	return StatusOK, nil
}

// recoverFaultyLogs is the leader-side CTRL driver. The caller must already
// have detected a CorruptionAwareLogStore. It must run on the main thread,
// after winning an election and before advertising leadership.
func (r *Raft) recoverFaultyLogs(store CorruptionAwareLogStore) error {
	faulty, err := store.GetFaultyEntries()
	if err != nil {
		return fmt.Errorf("list faulty entries: %w", err)
	}
	if len(faulty) == 0 {
		return nil
	}

	if _, ok := r.trans.(WithRecovery); !ok {
		r.logger.Warn("faulty log entries present but transport does not support RecoverEntry; skipping recovery",
			"count", len(faulty))
		return nil
	}

	quorum := r.quorumSize()
	r.logger.Info("recovering faulty log entries", "count", len(faulty), "quorum", quorum)

	for {
		faulty, err = store.GetFaultyEntries()
		if err != nil {
			return fmt.Errorf("list faulty entries: %w", err)
		}
		if len(faulty) == 0 {
			return nil
		}
		idx := faulty[0].Index
		if err := r.recoverOneFaulty(store, faulty[0], quorum); err != nil {
			return err
		}
		remaining, err := store.GetFaultyEntries()
		if err != nil {
			return fmt.Errorf("list faulty entries: %w", err)
		}
		if len(remaining) > 0 && remaining[0].Index == idx {
			return fmt.Errorf("log recovery made no progress at index %d", idx)
		}
	}
}

func (r *Raft) recoverOneFaulty(store CorruptionAwareLogStore, fe FaultyEntry, quorum int) error {
	timeout := r.config().ElectionTimeout
	var outcome recoverOutcome
	var replica *Log
	for attempt := 1; attempt <= recoverMaxAttempts; attempt++ {
		have, dontHave, entry := r.queryVoters(fe.Index, fe.Term, timeout)
		outcome = recoverDecision(have, dontHave, quorum)
		replica = entry
		r.logger.Info("recovery tally",
			"index", fe.Index, "term", fe.Term,
			"have", have, "dontHave", dontHave, "quorum", quorum,
			"outcome", outcome, "attempt", attempt)
		if outcome != recoverAmbiguous {
			break
		}
		if attempt < recoverMaxAttempts {
			select {
			case <-time.After(timeout):
			case <-r.shutdownCh:
				return ErrRaftShutdown
			}
		}
	}

	switch outcome {
	case recoverRepair:
		if replica == nil {
			return fmt.Errorf("majority have entry %d but no copy was returned", fe.Index)
		}
		if err := store.RepairEntry(replica); err != nil {
			return fmt.Errorf("repair entry %d: %w", fe.Index, err)
		}
		r.logger.Info("repaired faulty log entry from peer", "index", fe.Index, "term", fe.Term)
		return nil

	case recoverDiscard:
		if fe.Index <= r.getCommitIndex() {
			return fmt.Errorf("majority dont-have for index %d which is at or below commit index %d",
				fe.Index, r.getCommitIndex())
		}
		lastIdx := r.getLastIndex()
		if err := r.logs.DeleteRange(fe.Index, lastIdx); err != nil {
			return fmt.Errorf("truncate uncommitted suffix from %d: %w", fe.Index, err)
		}
		if r.configurations.latestIndex >= fe.Index {
			r.setLatestConfiguration(r.configurations.committed, r.configurations.committedIndex)
		}
		newLast, err := r.logs.LastIndex()
		if err != nil {
			return err
		}
		var lastLog Log
		if newLast > 0 {
			if err := r.logs.GetLog(newLast, &lastLog); err != nil {
				return fmt.Errorf("read new last log %d after truncate: %w", newLast, err)
			}
			r.setLastLog(lastLog.Index, lastLog.Term)
		} else {
			r.setLastLog(0, 0)
		}
		r.logger.Info("discarded uncommitted faulty suffix", "from", fe.Index, "to", lastIdx, "newLast", newLast)
		return nil

	default:
		return fmt.Errorf("%w: index %d term %d", ErrRecoveryAmbiguous, fe.Index, fe.Term)
	}
}

// queryVoters asks every other voter for RecoverEntry. This node's own
// HaveFaulty copy is not included in either tally.
func (r *Raft) queryVoters(index, term uint64, timeout time.Duration) (have, dontHave int, replica *Log) {
	recTrans, ok := r.trans.(WithRecovery)
	if !ok {
		return 0, 0, nil
	}

	type vote struct {
		resp RecoverEntryResponse
		err  error
	}

	var peers []Server
	for _, s := range r.configurations.latest.Servers {
		if s.Suffrage != Voter || s.ID == r.localID {
			continue
		}
		peers = append(peers, s)
	}
	if len(peers) == 0 {
		return 0, 0, nil
	}

	ch := make(chan vote, len(peers))
	args := &RecoverEntryRequest{
		RPCHeader: r.getRPCHeader(),
		Index:     index,
		Term:      term,
	}
	for _, p := range peers {
		go func(p Server) {
			var resp RecoverEntryResponse
			err := recTrans.RecoverEntry(p.ID, p.Address, args, &resp)
			ch <- vote{resp: resp, err: err}
		}(p)
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	received := 0
	for received < len(peers) {
		select {
		case v := <-ch:
			received++
			if v.err != nil {
				r.logger.Warn("RecoverEntry failed", "index", index, "error", v.err)
				continue
			}
			switch v.resp.Result {
			case RecoveryHave:
				have++
				if replica == nil && v.resp.Entry != nil {
					copy := *v.resp.Entry
					replica = &copy
				}
			case RecoveryDontHave:
				dontHave++
			}
		case <-timer.C:
			return have, dontHave, replica
		case <-r.shutdownCh:
			return have, dontHave, replica
		}
	}
	return have, dontHave, replica
}

func (o recoverOutcome) String() string {
	switch o {
	case recoverRepair:
		return "repair"
	case recoverDiscard:
		return "discard"
	default:
		return "ambiguous"
	}
}

var _ WithRecovery = (*NetworkTransport)(nil)
var _ WithRecovery = (*InmemTransport)(nil)
