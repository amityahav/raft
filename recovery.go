// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"errors"
	"fmt"
	"time"

	"github.com/hashicorp/go-metrics"
)

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

// recoverDecision applies the CTRL log-recovery rule (FAST'18 §3.4.3).
// HaveFaulty (including this node's own copy) counts for neither side.
//
//	≥1 Have            → repair (one intact copy is enough)
//	majority DontHave  → discard (entry was never committed)
//	otherwise          → ambiguous (wait / all remaining copies faulty)
func recoverDecision(have, dontHave, quorum int) recoverOutcome {
	if have >= 1 {
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

// voterPeers returns every voter other than this node from the latest
// configuration. It reads r.configurations and must be called on the main
// thread (e.g. in runLeader); the resulting snapshot can then be handed to a
// background worker.
func (r *Raft) voterPeers() []Server {
	var peers []Server
	for _, s := range r.configurations.latest.Servers {
		if s.Suffrage != Voter || s.ID == r.localID {
			continue
		}
		peers = append(peers, s)
	}
	return peers
}

// queryVoters asks each given voter for RecoverEntry. This node's own
// HaveFaulty copy is not included in either tally. The peers slice is a
// snapshot so this is safe to call from a background goroutine.
func (r *Raft) queryVoters(peers []Server, index, term uint64, timeout time.Duration) (have, dontHave int, replica *Log) {
	if len(peers) == 0 {
		return 0, 0, nil
	}

	type vote struct {
		resp RecoverEntryResponse
		err  error
	}

	ch := make(chan vote, len(peers))
	args := &RecoverEntryRequest{
		RPCHeader: r.getRPCHeader(),
		Index:     index,
		Term:      term,
	}
	trans, ok := r.trans.(WithRecovery)
	if !ok {
		r.logger.Error("CTRL enabled but transport does not support RecoverEntry")
		return 0, 0, nil
	}
	for _, p := range peers {
		go func(p Server) {
			var resp RecoverEntryResponse
			err := trans.RecoverEntry(p.ID, p.Address, args, &resp)
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
