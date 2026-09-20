// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
)

const (
	// redundantSuffix is appended to key names to create the redundant copy.
	redundantSuffix = ".__redundant"

	// crc32Len is the byte length of a CRC32 checksum.
	crc32Len = 4
)

var (
	// ErrBothCopiesCorrupted is returned when both copies of a value in the
	// RedundantStableStore are corrupted and the original value cannot be recovered.
	ErrBothCopiesCorrupted = errors.New("both copies of stable store value are corrupted")

	// ErrCopiesDisagree is returned when both copies pass checksum verification
	// but contain different values. This indicates a bug, not a storage fault.
	ErrCopiesDisagree = errors.New("stable store redundant copies have divergent values")
)

// RedundantStableStore wraps a StableStore with dual-copy writes to protect
// critical metadata (CurrentTerm, LastVoteTerm, LastVoteCand) from storage
// corruption. Each value is stored twice under different keys with independent
// CRC32 checksums. On read, both copies are verified and the valid one is
// returned; if one copy is corrupted, the other is used transparently.
//
// This implements the metainfo redundancy described in §3.3.1 of the CTRL
// paper. The metainfo is special because it cannot be recovered from other
// nodes — it contains node-local state (e.g., current term, vote). Storing
// two local copies is cheap since metainfo is only a few tens of bytes and
// is updated infrequently.
type RedundantStableStore struct {
	inner StableStore
}

// NewRedundantStableStore creates a new RedundantStableStore wrapping the
// given StableStore. The wrapper is transparent — it satisfies the StableStore
// interface and can be used as a drop-in replacement.
func NewRedundantStableStore(inner StableStore) *RedundantStableStore {
	return &RedundantStableStore{inner: inner}
}

// Set stores a key-value pair in both the primary and redundant locations,
// each protected by a CRC32 checksum.
func (r *RedundantStableStore) Set(key []byte, val []byte) error {
	protected := appendCRC(val)

	if err := r.inner.Set(key, protected); err != nil {
		return fmt.Errorf("failed to write primary copy: %w", err)
	}
	if err := r.inner.Set(redundantKey(key), protected); err != nil {
		return fmt.Errorf("failed to write redundant copy: %w", err)
	}
	return nil
}

// Get retrieves a value by reading both copies and returning a valid one.
// If one copy is corrupted, the other is returned. If both are corrupted,
// ErrBothCopiesCorrupted is returned.
func (r *RedundantStableStore) Get(key []byte) ([]byte, error) {
	primary, primaryErr := r.inner.Get(key)
	secondary, secondaryErr := r.inner.Get(redundantKey(key))

	primaryVal, primaryOK := verifyCRC(primary, primaryErr)
	secondaryVal, secondaryOK := verifyCRC(secondary, secondaryErr)

	switch {
	case primaryOK && secondaryOK:
		if !bytesEqual(primaryVal, secondaryVal) {
			return nil, ErrCopiesDisagree
		}
		return primaryVal, nil

	case primaryOK:
		// Secondary corrupted or missing — repair it.
		_ = r.inner.Set(redundantKey(key), appendCRC(primaryVal))
		return primaryVal, nil

	case secondaryOK:
		// Primary corrupted or missing — repair it.
		_ = r.inner.Set(key, appendCRC(secondaryVal))
		return secondaryVal, nil

	default:
		// Both are either corrupted, missing, or errored.
		// If both are "not found", propagate that as-is.
		if isNotFound(primaryErr) && isNotFound(secondaryErr) {
			return nil, primaryErr
		}
		return nil, ErrBothCopiesCorrupted
	}
}

// SetUint64 stores a uint64 value in both copies.
func (r *RedundantStableStore) SetUint64(key []byte, val uint64) error {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, val)
	return r.Set(key, buf)
}

// GetUint64 retrieves a uint64 value using the redundancy-verified Get.
func (r *RedundantStableStore) GetUint64(key []byte) (uint64, error) {
	val, err := r.Get(key)
	if err != nil {
		if isNotFound(err) {
			return 0, nil
		}
		return 0, err
	}
	if len(val) != 8 {
		return 0, fmt.Errorf("invalid uint64 value length: %d", len(val))
	}
	return binary.BigEndian.Uint64(val), nil
}

// VerifyIntegrity checks all keys in the store for integrity by attempting
// to read and verify both copies. This is a diagnostic tool — it returns
// a list of keys whose copies are corrupted or disagree.
//
// Note: this method can only check keys that are known. Since StableStore
// does not provide a key-enumeration method, this checks the well-known
// raft metadata keys.
func (r *RedundantStableStore) VerifyIntegrity() []string {
	knownKeys := [][]byte{
		keyCurrentTerm,
		keyLastVoteTerm,
		keyLastVoteCand,
	}

	var corrupted []string
	for _, key := range knownKeys {
		if _, err := r.Get(key); err != nil && !isNotFound(err) {
			corrupted = append(corrupted, string(key))
		}
	}
	return corrupted
}

// appendCRC returns a new slice containing val followed by its CRC32 checksum.
func appendCRC(val []byte) []byte {
	checksum := crc32.ChecksumIEEE(val)
	protected := make([]byte, len(val)+crc32Len)
	copy(protected, val)
	binary.BigEndian.PutUint32(protected[len(val):], checksum)
	return protected
}

// verifyCRC validates a protected value (data + CRC32 suffix). Returns the
// original data and true if valid, or nil and false if corrupted or on error.
func verifyCRC(protected []byte, readErr error) ([]byte, bool) {
	if readErr != nil {
		return nil, false
	}
	if len(protected) < crc32Len {
		return nil, false
	}

	data := protected[:len(protected)-crc32Len]
	stored := binary.BigEndian.Uint32(protected[len(protected)-crc32Len:])
	computed := crc32.ChecksumIEEE(data)

	if stored != computed {
		return nil, false
	}
	return data, true
}

// redundantKey returns the key used for the redundant copy.
func redundantKey(key []byte) []byte {
	rk := make([]byte, len(key)+len(redundantSuffix))
	copy(rk, key)
	copy(rk[len(key):], redundantSuffix)
	return rk
}

// bytesEqual compares two byte slices for equality.
func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// isNotFound checks if an error represents a "not found" condition.
// The StableStore interface returns an error with message "not found"
// when a key does not exist.
func isNotFound(err error) bool {
	return err != nil && err.Error() == "not found"
}
