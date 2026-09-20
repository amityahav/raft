// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import "fmt"

// IntegrityStatus describes the integrity state of a persistent data item
// as determined by the local storage layer (CLSTORE). It is used by the
// distributed recovery protocol to decide how to handle each item.
type IntegrityStatus int

const (
	// StatusOK indicates the data item passed integrity verification.
	StatusOK IntegrityStatus = iota

	// StatusCorrupted indicates the data item failed its checksum
	// verification but a persist record confirms the item was previously
	// written successfully. This distinguishes genuine storage corruption
	// from crash-induced partial writes.
	StatusCorrupted

	// StatusInaccessible indicates the data item could not be read due
	// to an I/O error (e.g., EIO from the underlying storage device).
	StatusInaccessible
)

// String returns a human-readable representation of the IntegrityStatus.
func (s IntegrityStatus) String() string {
	switch s {
	case StatusOK:
		return "OK"
	case StatusCorrupted:
		return "Corrupted"
	case StatusInaccessible:
		return "Inaccessible"
	default:
		return fmt.Sprintf("IntegrityStatus(%d)", int(s))
	}
}

// IsFaulty returns true if the status indicates the data is not usable.
func (s IntegrityStatus) IsFaulty() bool {
	return s != StatusOK
}

// FaultyEntry identifies a log entry whose data has been detected as
// corrupted or inaccessible by the local storage layer. The distributed
// recovery protocol uses this information to fetch correct copies from
// peers.
type FaultyEntry struct {
	// Index is the log index of the faulty entry.
	Index uint64

	// Term is the election term of the faulty entry, read from the
	// physically-separated identifier. The ⟨Term, Index⟩ pair uniquely
	// identifies a log entry across the cluster.
	Term uint64

	// Status describes the type of fault detected.
	Status IntegrityStatus
}

// RecoveryResponse represents a follower's response when the leader queries
// for a specific log entry during distributed log recovery. The leader
// collects these responses to determine whether a faulty entry is committed
// (and must be recovered) or uncommitted (and can be safely discarded).
type RecoveryResponse int

const (
	// RecoveryHave indicates the follower has the requested entry
	// ⟨term, index⟩ and it is not faulty. The response includes the
	// correct entry data that the leader can use to repair its copy.
	RecoveryHave RecoveryResponse = iota

	// RecoveryDontHave indicates the follower does not have an entry
	// at the requested ⟨term, index⟩. If a majority of followers
	// respond with DontHave, the leader can conclude the entry was
	// uncommitted and safely discard it.
	RecoveryDontHave

	// RecoveryHaveFaulty indicates the follower has the requested
	// entry ⟨term, index⟩ but it is also faulty on this follower.
	// The leader must wait for other responses to make a decision.
	RecoveryHaveFaulty
)

// String returns a human-readable representation of the RecoveryResponse.
func (r RecoveryResponse) String() string {
	switch r {
	case RecoveryHave:
		return "Have"
	case RecoveryDontHave:
		return "DontHave"
	case RecoveryHaveFaulty:
		return "HaveFaulty"
	default:
		return fmt.Sprintf("RecoveryResponse(%d)", int(r))
	}
}

// FaultySnapshotChunk identifies a snapshot chunk whose data has been
// detected as corrupted or inaccessible by the local storage layer.
type FaultySnapshotChunk struct {
	// SnapshotID identifies the snapshot containing the faulty chunk.
	SnapshotID string

	// ChunkIndex is the zero-based index of the faulty chunk within
	// the snapshot.
	ChunkIndex int

	// Status describes the type of fault detected.
	Status IntegrityStatus
}

// CorruptionAwareLogStore extends LogStore with the ability to detect
// corrupted log entries, distinguish crashes from corruption, and repair
// faulty entries using data recovered from peers.
//
// This interface follows the optional extension pattern used by
// MonotonicLogStore and CommitTrackingLogStore: implementations that
// satisfy this interface enable CTRL (corruption-tolerant replication)
// features. The raft core detects this interface via type assertion and
// falls back to standard behavior when it is not available.
//
// The design is based on the CLSTORE local storage layer described in
// "Protocol-Aware Recovery for Consensus-Based Storage" (FAST'18).
type CorruptionAwareLogStore interface {
	LogStore

	// GetLogWithIntegrity reads a log entry at the given index and
	// reports its integrity status. Unlike GetLog, a corrupted entry
	// does NOT cause an error return — instead the status is set to
	// StatusCorrupted and the Log fields that could be read (e.g.,
	// Term and Index from the separated identifier) are populated.
	//
	// Returns ErrLogNotFound if the index is not in the store's range.
	// Returns a non-nil error only for problems unrelated to data
	// integrity (e.g., store is closed).
	GetLogWithIntegrity(index uint64, log *Log) (IntegrityStatus, error)

	// GetFaultyEntries returns all log entries currently known to be
	// faulty (corrupted or inaccessible). This is called by the
	// distributed recovery protocol to learn which entries need to be
	// recovered from peers.
	//
	// The returned slice is a snapshot of the current faulty set;
	// entries that have been repaired via RepairEntry are excluded.
	GetFaultyEntries() ([]FaultyEntry, error)

	// RepairEntry overwrites a faulty log entry with correct data
	// received from a peer. The entry's Index is used to locate the
	// slot to overwrite. After a successful repair, the entry is
	// removed from the faulty set and subsequent GetLogWithIntegrity
	// calls for that index will return StatusOK.
	//
	// Returns an error if the entry at the given index is not currently
	// marked as faulty, or if the write fails.
	RepairEntry(log *Log) error

	// DisentangleCrashCorruption scans the log to separate entries
	// that are faulty due to crash-induced partial writes from entries
	// that are genuinely corrupted by storage faults.
	//
	// Crash-induced partial writes (entry written but persist record
	// absent) are safe to discard — the node never acknowledged them.
	// Genuine corruptions (persist record present but data mismatches
	// checksum) must be reported for distributed recovery.
	//
	// This method is intended to be called once during store
	// initialization (recovery on open). It returns:
	//   - lastSafeIndex: the index of the last entry that is either
	//     intact or genuinely corrupted (i.e., the point after which
	//     crash-induced partial writes were discarded).
	//   - faultyEntries: entries that are genuinely corrupted and
	//     require distributed recovery.
	//
	// After this call, the store's faulty set is populated with the
	// returned faultyEntries.
	DisentangleCrashCorruption() (lastSafeIndex uint64, faultyEntries []FaultyEntry, err error)
}

// ChunkedSnapshotStore extends SnapshotStore with chunk-level integrity
// verification and repair capabilities. This enables fine-grained
// snapshot recovery: instead of transferring an entire snapshot when
// corruption is detected, only the faulty chunks need to be fetched
// from peers.
//
// The design requires that snapshots across nodes are byte-identical
// at the same snapshot index. This is achieved by leader-initiated
// identical snapshots (implemented in Phase 2), where all nodes take
// a snapshot at the same log index via a log marker entry.
//
// Chunk size is implementation-defined but typically 4096 bytes to
// match filesystem block size, balancing recovery granularity against
// metadata overhead.
type ChunkedSnapshotStore interface {
	SnapshotStore

	// OpenChunk reads a single chunk from the snapshot identified by
	// id. chunkIndex is zero-based. Returns the chunk data or an error
	// if the chunk cannot be read.
	OpenChunk(id string, chunkIndex int) ([]byte, error)

	// GetChunkIntegrity verifies the integrity of a single chunk
	// within a snapshot. The chunk's data is checked against its
	// stored checksum.
	GetChunkIntegrity(id string, chunkIndex int) (IntegrityStatus, error)

	// RepairChunk overwrites a faulty chunk with correct data received
	// from a peer and updates the chunk's checksum. Returns an error
	// if the write fails.
	RepairChunk(id string, chunkIndex int, data []byte) error

	// ChunkCount returns the number of chunks in the snapshot
	// identified by id. This is used to iterate over all chunks
	// for integrity verification.
	ChunkCount(id string) (int, error)

	// GetFaultyChunks returns all snapshot chunks currently known to
	// be faulty across all snapshots in the store.
	GetFaultyChunks() ([]FaultySnapshotChunk, error)
}
