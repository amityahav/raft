// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"errors"
	"fmt"
	"time"

	"github.com/hashicorp/go-metrics"
)

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

var _ WithRecovery = (*NetworkTransport)(nil)
var _ WithRecovery = (*InmemTransport)(nil)
