// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

// newChunkedStore returns a ChunkedFileSnapshotStore with a small chunk size
// so tests exercise multi-chunk behavior with modest payloads.
func newChunkedStore(t *testing.T, chunkSize int) *ChunkedFileSnapshotStore {
	t.Helper()
	dir := t.TempDir()
	store, err := NewChunkedFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	require.NoError(t, err)
	store.chunkSize = chunkSize
	return store
}

// writeChunkedSnapshot persists data as a snapshot and returns its id.
func writeChunkedSnapshot(t *testing.T, store *ChunkedFileSnapshotStore, data []byte) string {
	t.Helper()
	_, trans := NewInmemTransport(NewInmemAddr())
	sink, err := store.Create(SnapshotVersionMax, 10, 3, Configuration{}, 0, trans)
	require.NoError(t, err)
	if len(data) > 0 {
		n, err := sink.Write(data)
		require.NoError(t, err)
		require.Equal(t, len(data), n)
	}
	require.NoError(t, sink.Close())
	return sink.ID()
}

// corruptStateByte flips a byte in a snapshot's state.bin at the given offset.
func corruptStateByte(t *testing.T, store *ChunkedFileSnapshotStore, id string, off int64) {
	t.Helper()
	statePath := filepath.Join(store.chunkDir(id), stateFilePath)
	fh, err := os.OpenFile(statePath, os.O_RDWR, 0o644)
	require.NoError(t, err)
	defer fh.Close()
	var b [1]byte
	_, err = fh.ReadAt(b[:], off)
	require.NoError(t, err)
	b[0] ^= 0xFF
	_, err = fh.WriteAt(b[:], off)
	require.NoError(t, err)
}

func makeData(n int) []byte {
	data := make([]byte, n)
	for i := range data {
		data[i] = byte(i % 251)
	}
	return data
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

func TestChunkedSnapshot_InterfaceAssertions(t *testing.T) {
	var impl interface{} = &ChunkedFileSnapshotStore{}
	_, ok := impl.(SnapshotStore)
	assert.True(t, ok, "must be a SnapshotStore")
	_, ok = impl.(ChunkedSnapshotStore)
	assert.True(t, ok, "must be a ChunkedSnapshotStore")
}

func TestChunkedSnapshot_RoundTrip(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(40) // 3 chunks: 16 + 16 + 8

	id := writeChunkedSnapshot(t, store, data)

	count, err := store.ChunkCount(id)
	require.NoError(t, err)
	assert.Equal(t, 3, count)

	// Reassemble via OpenChunk and compare to the original.
	var buf bytes.Buffer
	for i := 0; i < count; i++ {
		chunk, err := store.OpenChunk(id, i)
		require.NoError(t, err)
		buf.Write(chunk)
		status, err := store.GetChunkIntegrity(id, i)
		require.NoError(t, err)
		assert.Equal(t, StatusOK, status, "chunk %d", i)
	}
	assert.Equal(t, data, buf.Bytes())

	// The last chunk is short.
	last, err := store.OpenChunk(id, count-1)
	require.NoError(t, err)
	assert.Len(t, last, 8)

	// The standard whole-snapshot read path still works.
	_, rc, err := store.Open(id)
	require.NoError(t, err)
	got, err := io.ReadAll(rc)
	require.NoError(t, err)
	require.NoError(t, rc.Close())
	assert.Equal(t, data, got)
}

func TestChunkedSnapshot_ExactMultipleNoShortChunk(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(32) // exactly 2 chunks

	id := writeChunkedSnapshot(t, store, data)

	count, err := store.ChunkCount(id)
	require.NoError(t, err)
	assert.Equal(t, 2, count)

	for i := 0; i < count; i++ {
		chunk, err := store.OpenChunk(id, i)
		require.NoError(t, err)
		assert.Len(t, chunk, 16)
	}

	_, err = store.OpenChunk(id, count) // out of range
	assert.Error(t, err)
}

func TestChunkedSnapshot_EmptyState(t *testing.T) {
	store := newChunkedStore(t, 16)
	id := writeChunkedSnapshot(t, store, nil)

	count, err := store.ChunkCount(id)
	require.NoError(t, err)
	assert.Equal(t, 0, count)

	faulty, err := store.GetFaultyChunks()
	require.NoError(t, err)
	assert.Empty(t, faulty)
}

func TestChunkedSnapshot_SingleChunkCorruption(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(40)
	id := writeChunkedSnapshot(t, store, data)

	// Corrupt a byte inside chunk 1 (offsets 16..31).
	corruptStateByte(t, store, id, 20)

	s0, err := store.GetChunkIntegrity(id, 0)
	require.NoError(t, err)
	assert.Equal(t, StatusOK, s0)

	s1, err := store.GetChunkIntegrity(id, 1)
	require.NoError(t, err)
	assert.Equal(t, StatusCorrupted, s1)

	s2, err := store.GetChunkIntegrity(id, 2)
	require.NoError(t, err)
	assert.Equal(t, StatusOK, s2)

	faulty, err := store.GetFaultyChunks()
	require.NoError(t, err)
	require.Len(t, faulty, 1)
	assert.Equal(t, id, faulty[0].SnapshotID)
	assert.Equal(t, 1, faulty[0].ChunkIndex)
	assert.Equal(t, StatusCorrupted, faulty[0].Status)
}

func TestChunkedSnapshot_RepairChunk(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(40)
	id := writeChunkedSnapshot(t, store, data)

	// The correct bytes for chunk 1 (as a peer would supply them).
	original := append([]byte(nil), data[16:32]...)

	corruptStateByte(t, store, id, 20)
	status, err := store.GetChunkIntegrity(id, 1)
	require.NoError(t, err)
	require.Equal(t, StatusCorrupted, status)

	// Whole-snapshot read fails while corrupted.
	_, _, err = store.Open(id)
	require.Error(t, err)

	require.NoError(t, store.RepairChunk(id, 1, original))

	status, err = store.GetChunkIntegrity(id, 1)
	require.NoError(t, err)
	assert.Equal(t, StatusOK, status)

	// Restoring the exact bytes makes the whole-file CRC64 valid again.
	_, rc, err := store.Open(id)
	require.NoError(t, err)
	got, err := io.ReadAll(rc)
	require.NoError(t, err)
	require.NoError(t, rc.Close())
	assert.Equal(t, data, got)
}

func TestChunkedSnapshot_RepairShortLastChunk(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(40) // last chunk is 8 bytes
	id := writeChunkedSnapshot(t, store, data)

	original := append([]byte(nil), data[32:40]...)
	corruptStateByte(t, store, id, 36)

	status, err := store.GetChunkIntegrity(id, 2)
	require.NoError(t, err)
	require.Equal(t, StatusCorrupted, status)

	require.NoError(t, store.RepairChunk(id, 2, original))
	status, err = store.GetChunkIntegrity(id, 2)
	require.NoError(t, err)
	assert.Equal(t, StatusOK, status)
}

func TestChunkedSnapshot_RepairMismatchRejected(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(40)
	id := writeChunkedSnapshot(t, store, data)

	corruptStateByte(t, store, id, 20)

	// Wrong-length replacement is rejected.
	err := store.RepairChunk(id, 1, []byte("too short"))
	assert.ErrorIs(t, err, ErrChunkRepairMismatch)

	// Right length but wrong bytes is rejected.
	wrong := makeData(16)
	err = store.RepairChunk(id, 1, wrong)
	assert.ErrorIs(t, err, ErrChunkRepairMismatch)

	// The chunk is still corrupted (repair did not write bad data).
	status, err := store.GetChunkIntegrity(id, 1)
	require.NoError(t, err)
	assert.Equal(t, StatusCorrupted, status)
}

func TestChunkedSnapshot_CorruptSidecarDetected(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(40)
	id := writeChunkedSnapshot(t, store, data)

	// Corrupt the chunks.meta sidecar so its self-CRC no longer matches.
	metaPath := filepath.Join(store.chunkDir(id), chunkMetaFilePath)
	raw, err := os.ReadFile(metaPath)
	require.NoError(t, err)
	// Flip a digit inside one of the CRC values.
	raw = bytes.Replace(raw, []byte("\"crcs\":["), []byte("\"crcs\":[1,"), 1)
	require.NoError(t, os.WriteFile(metaPath, raw, 0o644))

	_, err = store.ChunkCount(id)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "corrupted")
}

func TestChunkedSnapshot_DataCorruptionDoesNotTouchSidecar(t *testing.T) {
	store := newChunkedStore(t, 16)
	data := makeData(40)
	id := writeChunkedSnapshot(t, store, data)

	// Corrupting state.bin must not affect the physically-separate sidecar:
	// the checksums remain intact and correctly flag the bad chunk.
	corruptStateByte(t, store, id, 20)

	count, err := store.ChunkCount(id)
	require.NoError(t, err)
	assert.Equal(t, 3, count, "sidecar still readable after data corruption")

	status, err := store.GetChunkIntegrity(id, 1)
	require.NoError(t, err)
	assert.Equal(t, StatusCorrupted, status)
}

// A snapshot lacking a chunks.meta sidecar (e.g., not produced by this store)
// is rejected rather than regenerated. The store assumes it is used from the
// start on a fresh Raft instance.
func TestChunkedSnapshot_MissingSidecarRejected(t *testing.T) {
	dir := t.TempDir()

	// Write a snapshot using the plain store — no sidecar is written.
	plain, err := NewFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	require.NoError(t, err)
	_, trans := NewInmemTransport(NewInmemAddr())
	sink, err := plain.Create(SnapshotVersionMax, 10, 3, Configuration{}, 0, trans)
	require.NoError(t, err)
	_, err = sink.Write(makeData(40))
	require.NoError(t, err)
	require.NoError(t, sink.Close())
	id := sink.ID()

	chunked, err := NewChunkedFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	require.NoError(t, err)
	chunked.chunkSize = 16

	metaPath := filepath.Join(chunked.chunkDir(id), chunkMetaFilePath)
	_, statErr := os.Stat(metaPath)
	require.True(t, os.IsNotExist(statErr), "sidecar should not exist")

	_, err = chunked.ChunkCount(id)
	require.Error(t, err, "missing sidecar must be an error")

	// It must not be lazily generated as a side effect.
	_, statErr = os.Stat(metaPath)
	assert.True(t, os.IsNotExist(statErr), "sidecar must not be generated")
}

func TestChunkedSnapshot_MultipleSnapshotsFaultyScan(t *testing.T) {
	store := newChunkedStore(t, 16)

	id1 := writeChunkedSnapshot(t, store, makeData(40))
	id2 := writeChunkedSnapshot(t, store, makeData(48))

	corruptStateByte(t, store, id1, 4)  // chunk 0 of id1
	corruptStateByte(t, store, id2, 40) // chunk 2 of id2

	faulty, err := store.GetFaultyChunks()
	require.NoError(t, err)
	require.Len(t, faulty, 2)

	found := map[string]int{}
	for _, fc := range faulty {
		found[fc.SnapshotID] = fc.ChunkIndex
		assert.Equal(t, StatusCorrupted, fc.Status)
	}
	assert.Equal(t, 0, found[id1])
	assert.Equal(t, 2, found[id2])
}
