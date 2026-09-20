// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRedundantStableStore_BasicSetGet(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	// Set and Get a byte value.
	require.NoError(t, store.Set([]byte("key1"), []byte("value1")))

	val, err := store.Get([]byte("key1"))
	require.NoError(t, err)
	assert.Equal(t, []byte("value1"), val)
}

func TestRedundantStableStore_BasicSetGetUint64(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.SetUint64([]byte("term"), 42))

	val, err := store.GetUint64([]byte("term"))
	require.NoError(t, err)
	assert.Equal(t, uint64(42), val)
}

func TestRedundantStableStore_GetMissing(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	// Byte get for missing key returns "not found" error (pass-through).
	_, err := store.Get([]byte("nonexistent"))
	require.Error(t, err)
	assert.True(t, isNotFound(err))

	// Uint64 get for missing key returns 0, nil (matches StableStore contract).
	val, err := store.GetUint64([]byte("nonexistent"))
	require.NoError(t, err)
	assert.Equal(t, uint64(0), val)
}

func TestRedundantStableStore_PrimaryCorrupted(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.Set([]byte("key1"), []byte("hello")))

	// Corrupt the primary copy by flipping a byte.
	corruptInnerValue(t, inner, []byte("key1"))

	// Get should still succeed — recovered from secondary.
	val, err := store.Get([]byte("key1"))
	require.NoError(t, err)
	assert.Equal(t, []byte("hello"), val)

	// The primary should have been auto-repaired.
	val2, err := store.Get([]byte("key1"))
	require.NoError(t, err)
	assert.Equal(t, []byte("hello"), val2)
}

func TestRedundantStableStore_SecondaryCorrupted(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.Set([]byte("key1"), []byte("world")))

	// Corrupt the secondary (redundant) copy.
	corruptInnerValue(t, inner, redundantKey([]byte("key1")))

	// Get should still succeed — uses primary.
	val, err := store.Get([]byte("key1"))
	require.NoError(t, err)
	assert.Equal(t, []byte("world"), val)
}

func TestRedundantStableStore_BothCorrupted(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.Set([]byte("key1"), []byte("data")))

	// Corrupt both copies.
	corruptInnerValue(t, inner, []byte("key1"))
	corruptInnerValue(t, inner, redundantKey([]byte("key1")))

	_, err := store.Get([]byte("key1"))
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrBothCopiesCorrupted))
}

func TestRedundantStableStore_Uint64PrimaryCorrupted(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.SetUint64([]byte("term"), 99))

	// Corrupt the primary.
	corruptInnerValue(t, inner, []byte("term"))

	val, err := store.GetUint64([]byte("term"))
	require.NoError(t, err)
	assert.Equal(t, uint64(99), val)
}

func TestRedundantStableStore_Uint64BothCorrupted(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.SetUint64([]byte("term"), 99))

	corruptInnerValue(t, inner, []byte("term"))
	corruptInnerValue(t, inner, redundantKey([]byte("term")))

	_, err := store.GetUint64([]byte("term"))
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrBothCopiesCorrupted))
}

func TestRedundantStableStore_OverwritePreservesRedundancy(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.SetUint64([]byte("term"), 1))
	require.NoError(t, store.SetUint64([]byte("term"), 2))

	val, err := store.GetUint64([]byte("term"))
	require.NoError(t, err)
	assert.Equal(t, uint64(2), val)

	// Corrupt primary — should recover the latest value from secondary.
	corruptInnerValue(t, inner, []byte("term"))

	val, err = store.GetUint64([]byte("term"))
	require.NoError(t, err)
	assert.Equal(t, uint64(2), val)
}

func TestRedundantStableStore_VerifyIntegrity_AllClean(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.SetUint64(keyCurrentTerm, 5))
	require.NoError(t, store.SetUint64(keyLastVoteTerm, 4))
	require.NoError(t, store.Set(keyLastVoteCand, []byte("node-1")))

	corrupted := store.VerifyIntegrity()
	assert.Empty(t, corrupted)
}

func TestRedundantStableStore_VerifyIntegrity_DetectsCorruption(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.SetUint64(keyCurrentTerm, 5))
	require.NoError(t, store.SetUint64(keyLastVoteTerm, 4))
	require.NoError(t, store.Set(keyLastVoteCand, []byte("node-1")))

	// Corrupt both copies of CurrentTerm.
	corruptInnerValue(t, inner, keyCurrentTerm)
	corruptInnerValue(t, inner, redundantKey(keyCurrentTerm))

	corrupted := store.VerifyIntegrity()
	assert.Contains(t, corrupted, string(keyCurrentTerm))
	assert.Len(t, corrupted, 1)
}

func TestRedundantStableStore_AutoRepairOnRead(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.Set([]byte("key"), []byte("val")))

	// Corrupt primary.
	corruptInnerValue(t, inner, []byte("key"))

	// First read should recover from secondary and repair primary.
	val, err := store.Get([]byte("key"))
	require.NoError(t, err)
	assert.Equal(t, []byte("val"), val)

	// Now corrupt secondary — primary should have been repaired above.
	corruptInnerValue(t, inner, redundantKey([]byte("key")))

	// Should still work because primary was repaired.
	val, err = store.Get([]byte("key"))
	require.NoError(t, err)
	assert.Equal(t, []byte("val"), val)
}

func TestRedundantStableStore_EmptyValue(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	// Empty byte slice is a valid value.
	require.NoError(t, store.Set([]byte("empty"), []byte{}))

	val, err := store.Get([]byte("empty"))
	require.NoError(t, err)
	assert.Equal(t, []byte{}, val)
}

func TestRedundantStableStore_ConcurrentAccess(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	const goroutines = 10
	const iterations = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			key := []byte("concurrent-key")
			for i := 0; i < iterations; i++ {
				val := uint64(id*iterations + i)
				_ = store.SetUint64(key, val)
				_, _ = store.GetUint64(key)
			}
		}(g)
	}

	wg.Wait()

	// After all goroutines finish, a read should succeed without error.
	_, err := store.GetUint64([]byte("concurrent-key"))
	require.NoError(t, err)
}

func TestRedundantStableStore_PrimaryMissingSecondaryPresent(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	// Write normally.
	require.NoError(t, store.Set([]byte("key"), []byte("value")))

	// Simulate primary being deleted (missing file scenario from the
	// paper's FS metadata fault model).
	deleteInnerKey(inner, []byte("key"))

	// Get should recover from secondary.
	val, err := store.Get([]byte("key"))
	require.NoError(t, err)
	assert.Equal(t, []byte("value"), val)
}

func TestRedundantStableStore_TruncatedValue(t *testing.T) {
	inner := NewInmemStore()
	store := NewRedundantStableStore(inner)

	require.NoError(t, store.Set([]byte("key"), []byte("hello")))

	// Truncate the primary to fewer than crc32Len bytes.
	require.NoError(t, inner.Set([]byte("key"), []byte{0x01}))

	// Should recover from secondary.
	val, err := store.Get([]byte("key"))
	require.NoError(t, err)
	assert.Equal(t, []byte("hello"), val)
}

// --- helpers ---

// corruptInnerValue reads the raw value from the inner store and flips
// a bit in the data portion, causing a CRC mismatch.
func corruptInnerValue(t *testing.T, inner *InmemStore, key []byte) {
	t.Helper()
	raw, err := inner.Get(key)
	require.NoError(t, err)
	require.True(t, len(raw) > crc32Len, "value too short to corrupt")
	// Flip a bit in the first data byte.
	corrupted := make([]byte, len(raw))
	copy(corrupted, raw)
	corrupted[0] ^= 0xFF
	require.NoError(t, inner.Set(key, corrupted))
}

// deleteInnerKey removes a key directly from the inner InmemStore,
// simulating a missing file / metadata fault.
func deleteInnerKey(inner *InmemStore, key []byte) {
	inner.l.Lock()
	defer inner.l.Unlock()
	delete(inner.kv, string(key))
}

// --- verify helpers themselves ---

func TestAppendCRC_VerifyCRC_RoundTrip(t *testing.T) {
	data := []byte("test data for CRC")
	protected := appendCRC(data)

	val, ok := verifyCRC(protected, nil)
	require.True(t, ok)
	assert.Equal(t, data, val)
}

func TestVerifyCRC_DetectsCorruption(t *testing.T) {
	data := []byte("original data")
	protected := appendCRC(data)

	// Flip a byte.
	protected[2] ^= 0xFF

	_, ok := verifyCRC(protected, nil)
	assert.False(t, ok)
}

func TestVerifyCRC_RejectsShortValue(t *testing.T) {
	_, ok := verifyCRC([]byte{0x01, 0x02}, nil)
	assert.False(t, ok)
}

func TestVerifyCRC_RejectsOnError(t *testing.T) {
	_, ok := verifyCRC(nil, errors.New("I/O error"))
	assert.False(t, ok)
}

func TestVerifyCRC_EmptyData(t *testing.T) {
	// Empty data should still work — just a CRC of an empty slice.
	protected := appendCRC([]byte{})
	val, ok := verifyCRC(protected, nil)
	require.True(t, ok)
	assert.Equal(t, []byte{}, val)
}

func TestRedundantKey(t *testing.T) {
	key := []byte("CurrentTerm")
	rk := redundantKey(key)
	assert.Equal(t, "CurrentTerm.__redundant", string(rk))
}

func TestAppendCRC_Format(t *testing.T) {
	data := []byte("hello")
	protected := appendCRC(data)

	assert.Len(t, protected, len(data)+crc32Len)
	// The last 4 bytes should be the CRC32 of "hello".
	expected := crc32.ChecksumIEEE(data)
	actual := binary.BigEndian.Uint32(protected[len(data):])
	assert.Equal(t, expected, actual)
}
