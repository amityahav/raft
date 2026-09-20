// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

func testConfig(t *testing.T) (string, FileLogStoreConfig) {
	t.Helper()
	dir := t.TempDir()
	cfg := DefaultFileLogStoreConfig()
	cfg.NoSync = true // fast tests
	cfg.MaxEntriesPerSegment = 64
	cfg.SegmentSize = int64(segmentHeaderSize) + 64*int64(identifierSlotSize) + 256*1024 // small segments
	return dir, cfg
}

func testStore(t *testing.T) *FileLogStore {
	t.Helper()
	dir, cfg := testConfig(t)
	store, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	t.Cleanup(func() { store.Close() })
	return store
}

func testLogs(start, end uint64) []*Log {
	var logs []*Log
	for i := start; i <= end; i++ {
		logs = append(logs, &Log{
			Index:      i,
			Term:       1,
			Type:       LogCommand,
			Data:       []byte("test-data-" + string(rune('0'+i%10))),
			AppendedAt: time.Now(),
		})
	}
	return logs
}

// --------------------------------------------------------------------------
// Task 2b: Entry codec tests
// --------------------------------------------------------------------------

func TestEncodeDecodeLogEntry(t *testing.T) {
	original := &Log{
		Index:      42,
		Term:       7,
		Type:       LogCommand,
		Data:       []byte("hello world"),
		Extensions: []byte("ext-data"),
		AppendedAt: time.Now().Truncate(time.Millisecond),
	}

	encoded, err := encodeLogEntry(original)
	require.NoError(t, err)
	require.NotEmpty(t, encoded)

	decoded := &Log{}
	err = decodeLogEntry(encoded, decoded)
	require.NoError(t, err)

	assert.Equal(t, original.Index, decoded.Index)
	assert.Equal(t, original.Term, decoded.Term)
	assert.Equal(t, original.Type, decoded.Type)
	assert.Equal(t, original.Data, decoded.Data)
	assert.Equal(t, original.Extensions, decoded.Extensions)
}

func TestEncodeDecodeLogEntry_EmptyData(t *testing.T) {
	original := &Log{
		Index: 1,
		Term:  1,
		Type:  LogNoop,
	}

	encoded, err := encodeLogEntry(original)
	require.NoError(t, err)

	decoded := &Log{}
	err = decodeLogEntry(encoded, decoded)
	require.NoError(t, err)

	assert.Equal(t, original.Index, decoded.Index)
	assert.Equal(t, original.Term, decoded.Term)
	assert.Equal(t, original.Type, decoded.Type)
}

// --------------------------------------------------------------------------
// Task 2c: Identifier codec tests
// --------------------------------------------------------------------------

func TestEncodeDecodeIdentifier(t *testing.T) {
	rec := identifierRecord{
		Term:       5,
		Index:      100,
		DataOffset: 65536,
		DataLen:    1024,
	}

	buf := encodeIdentifier(rec)
	decoded, ok := decodeIdentifier(buf)

	require.True(t, ok)
	assert.Equal(t, rec.Term, decoded.Term)
	assert.Equal(t, rec.Index, decoded.Index)
	assert.Equal(t, rec.DataOffset, decoded.DataOffset)
	assert.Equal(t, rec.DataLen, decoded.DataLen)
}

func TestDecodeIdentifier_Empty(t *testing.T) {
	var buf [identifierSlotSize]byte
	_, ok := decodeIdentifier(buf)
	assert.False(t, ok, "all-zero slot should decode as empty")
}

func TestDecodeIdentifier_CorruptedCRC(t *testing.T) {
	rec := identifierRecord{Term: 1, Index: 1, DataOffset: 100, DataLen: 50}
	buf := encodeIdentifier(rec)

	// Corrupt the CRC.
	buf[31] ^= 0xFF
	_, ok := decodeIdentifier(buf)
	assert.False(t, ok, "corrupted CRC should fail decoding")
}

func TestDecodeIdentifier_CorruptedData(t *testing.T) {
	rec := identifierRecord{Term: 1, Index: 1, DataOffset: 100, DataLen: 50}
	buf := encodeIdentifier(rec)

	// Corrupt data (term field).
	buf[3] ^= 0xFF
	_, ok := decodeIdentifier(buf)
	assert.False(t, ok, "corrupted data should fail CRC check")
}

// --------------------------------------------------------------------------
// Task 2d-2j: Store integration tests
// --------------------------------------------------------------------------

func TestFileLogStore_EmptyStore(t *testing.T) {
	store := testStore(t)

	first, err := store.FirstIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(0), first)

	last, err := store.LastIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(0), last)

	var log Log
	err = store.GetLog(1, &log)
	assert.Equal(t, ErrLogNotFound, err)
}

func TestFileLogStore_StoreAndGet(t *testing.T) {
	store := testStore(t)

	logs := testLogs(1, 10)
	require.NoError(t, store.StoreLogs(logs))

	first, err := store.FirstIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(1), first)

	last, err := store.LastIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(10), last)

	for _, expected := range logs {
		var got Log
		require.NoError(t, store.GetLog(expected.Index, &got))
		assert.Equal(t, expected.Index, got.Index)
		assert.Equal(t, expected.Term, got.Term)
		assert.Equal(t, expected.Type, got.Type)
		assert.Equal(t, expected.Data, got.Data)
	}
}

func TestFileLogStore_StoreSingle(t *testing.T) {
	store := testStore(t)

	l := &Log{Index: 1, Term: 1, Type: LogCommand, Data: []byte("single")}
	require.NoError(t, store.StoreLog(l))

	var got Log
	require.NoError(t, store.GetLog(1, &got))
	assert.Equal(t, l.Data, got.Data)
}

func TestFileLogStore_GetLogOutOfRange(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	var log Log
	assert.Equal(t, ErrLogNotFound, store.GetLog(0, &log))
	assert.Equal(t, ErrLogNotFound, store.GetLog(6, &log))
	assert.Equal(t, ErrLogNotFound, store.GetLog(100, &log))
}

func TestFileLogStore_SegmentRotation(t *testing.T) {
	store := testStore(t)

	// MaxEntriesPerSegment is 64, so writing 150 entries should create 3 segments.
	require.NoError(t, store.StoreLogs(testLogs(1, 150)))

	assert.GreaterOrEqual(t, len(store.segments), 3)

	first, err := store.FirstIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(1), first)

	last, err := store.LastIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(150), last)

	// Verify all entries are readable across segments.
	for i := uint64(1); i <= 150; i++ {
		var log Log
		require.NoError(t, store.GetLog(i, &log), "failed to read entry %d", i)
		assert.Equal(t, i, log.Index)
	}
}

func TestFileLogStore_DeleteRange_FullSegments(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 130)))
	segCountBefore := len(store.segments)

	// Delete range covering the first full segment (entries 1-64).
	require.NoError(t, store.DeleteRange(1, 64))

	assert.Less(t, len(store.segments), segCountBefore)

	first, err := store.FirstIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(65), first)

	// Verify deleted entries are gone.
	var log Log
	assert.Equal(t, ErrLogNotFound, store.GetLog(1, &log))
	assert.Equal(t, ErrLogNotFound, store.GetLog(64, &log))

	// Verify remaining entries are intact.
	require.NoError(t, store.GetLog(65, &log))
	assert.Equal(t, uint64(65), log.Index)
}

func TestFileLogStore_DeleteRange_Partial(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 10)))

	// Delete entries 3-7.
	require.NoError(t, store.DeleteRange(3, 7))

	// Entries outside the range should still be readable.
	var log Log
	require.NoError(t, store.GetLog(1, &log))
	assert.Equal(t, uint64(1), log.Index)

	require.NoError(t, store.GetLog(2, &log))
	assert.Equal(t, uint64(2), log.Index)

	require.NoError(t, store.GetLog(8, &log))
	assert.Equal(t, uint64(8), log.Index)

	// Deleted entries should not be found (identifier cleared).
	for i := uint64(3); i <= 7; i++ {
		assert.Equal(t, ErrLogNotFound, store.GetLog(i, &log), "entry %d should be deleted", i)
	}
}

func TestFileLogStore_GetLogWithIntegrity_OK(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	var log Log
	status, err := store.GetLogWithIntegrity(3, &log)
	require.NoError(t, err)
	assert.Equal(t, StatusOK, status)
	assert.Equal(t, uint64(3), log.Index)
}

func TestFileLogStore_GetLogWithIntegrity_Corrupted(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	// Corrupt entry 3's data in the data region.
	seg := store.findSegment(3)
	require.NotNil(t, seg)
	rec := seg.index[3]

	// Flip a byte in the encoded data.
	corruptOff := rec.DataOffset + int64(entryLenSize) + 2
	var b [1]byte
	seg.file.ReadAt(b[:], corruptOff)
	b[0] ^= 0xFF
	seg.file.WriteAt(b[:], corruptOff)

	var log Log
	status, err := store.GetLogWithIntegrity(3, &log)
	require.NoError(t, err)
	assert.Equal(t, StatusCorrupted, status)
	// Term and Index should be populated from the identifier.
	assert.Equal(t, uint64(3), log.Index)
	assert.Equal(t, uint64(1), log.Term)
}

func TestFileLogStore_GetLog_CorruptedReturnsError(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	// Corrupt entry 2.
	seg := store.findSegment(2)
	require.NotNil(t, seg)
	rec := seg.index[2]
	corruptOff := rec.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	seg.file.ReadAt(b[:], corruptOff)
	b[0] ^= 0xFF
	seg.file.WriteAt(b[:], corruptOff)

	var log Log
	err := store.GetLog(2, &log)
	assert.Equal(t, ErrCorruptedEntry, err)
}

func TestFileLogStore_RepairEntry(t *testing.T) {
	store := testStore(t)

	logs := testLogs(1, 5)
	require.NoError(t, store.StoreLogs(logs))

	// Corrupt entry 4 and register it as faulty.
	seg := store.findSegment(4)
	require.NotNil(t, seg)
	rec := seg.index[4]
	corruptOff := rec.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	seg.file.ReadAt(b[:], corruptOff)
	b[0] ^= 0xFF
	seg.file.WriteAt(b[:], corruptOff)

	store.mu.Lock()
	store.faultySet[4] = FaultyEntry{Index: 4, Term: 1, Status: StatusCorrupted}
	store.mu.Unlock()

	// Repair with the same entry a peer would return for ⟨term 1, index 4⟩.
	correctLog := logs[3]
	require.NoError(t, store.RepairEntry(correctLog))

	// Verify it's no longer faulty.
	faulty, err := store.GetFaultyEntries()
	require.NoError(t, err)
	assert.Empty(t, faulty)

	// Verify the entry reads correctly now.
	var got Log
	require.NoError(t, store.GetLog(4, &got))
	assert.Equal(t, correctLog.Data, got.Data)
}

func TestFileLogStore_RepairEntry_NotFaulty(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	l := &Log{Index: 3, Term: 1, Type: LogCommand, Data: []byte("repair")}
	err := store.RepairEntry(l)
	assert.Equal(t, ErrEntryNotFaulty, err)
}

func TestFileLogStore_GetFaultyEntries_Sorted(t *testing.T) {
	store := testStore(t)

	store.faultySet[10] = FaultyEntry{Index: 10, Term: 2, Status: StatusCorrupted}
	store.faultySet[3] = FaultyEntry{Index: 3, Term: 1, Status: StatusCorrupted}
	store.faultySet[7] = FaultyEntry{Index: 7, Term: 2, Status: StatusInaccessible}

	faulty, err := store.GetFaultyEntries()
	require.NoError(t, err)
	require.Len(t, faulty, 3)
	assert.Equal(t, uint64(3), faulty[0].Index)
	assert.Equal(t, uint64(7), faulty[1].Index)
	assert.Equal(t, uint64(10), faulty[2].Index)
}

func TestFileLogStore_DisentangleCrashCorruption_CleanLog(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 10)))

	lastSafe, faulty, err := store.DisentangleCrashCorruption()
	require.NoError(t, err)
	assert.Equal(t, uint64(10), lastSafe)
	assert.Empty(t, faulty)
}

func TestFileLogStore_DisentangleCrashCorruption_CrashBoundary(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 10)))

	// Simulate a crash at entry 8: zero out its identifier slot.
	seg := store.active
	slot := uint32(7) // entry 8 is at slot 7 (0-based, baseIndex=1)
	var zeroBuf [identifierSlotSize]byte
	_, err := seg.file.WriteAt(zeroBuf[:], seg.slotOffset(slot))
	require.NoError(t, err)
	delete(seg.index, 8)

	lastSafe, faulty, err := store.DisentangleCrashCorruption()
	require.NoError(t, err)
	assert.Equal(t, uint64(7), lastSafe)
	assert.Empty(t, faulty, "crash entries should not appear as corrupted")

	// Entries 8-10 should be discarded.
	assert.Equal(t, uint32(7), seg.entryCount)

	last, err := store.LastIndex()
	require.NoError(t, err)
	// updateGlobalIndexes was called during construction; re-compute.
	store.mu.Lock()
	store.updateGlobalIndexes()
	store.mu.Unlock()
	last, _ = store.LastIndex()
	assert.Equal(t, uint64(7), last)
}

func TestFileLogStore_DisentangleCrashCorruption_CorruptedEntry(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 10)))

	// Corrupt entry 5's data (identifier remains intact).
	seg := store.active
	rec := seg.index[5]
	corruptOff := rec.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	seg.file.ReadAt(b[:], corruptOff)
	b[0] ^= 0xFF
	seg.file.WriteAt(b[:], corruptOff)

	store.faultySet = make(map[uint64]FaultyEntry) // reset
	lastSafe, faulty, err := store.DisentangleCrashCorruption()
	require.NoError(t, err)
	assert.Equal(t, uint64(10), lastSafe)
	require.Len(t, faulty, 1)
	assert.Equal(t, uint64(5), faulty[0].Index)
	assert.Equal(t, StatusCorrupted, faulty[0].Status)

	// The faulty set should contain the entry.
	assert.Contains(t, store.faultySet, uint64(5))
}

func TestFileLogStore_PersistAndReopen(t *testing.T) {
	dir, cfg := testConfig(t)

	// Create store, write entries, close.
	store1, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	require.NoError(t, store1.StoreLogs(testLogs(1, 20)))
	require.NoError(t, store1.Close())

	// Reopen.
	store2, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	defer store2.Close()

	first, err := store2.FirstIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(1), first)

	last, err := store2.LastIndex()
	require.NoError(t, err)
	assert.Equal(t, uint64(20), last)

	for i := uint64(1); i <= 20; i++ {
		var log Log
		require.NoError(t, store2.GetLog(i, &log), "entry %d not found after reopen", i)
		assert.Equal(t, i, log.Index)
	}
}

func TestFileLogStore_PersistAndReopen_MultiSegment(t *testing.T) {
	dir, cfg := testConfig(t)

	store1, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	require.NoError(t, store1.StoreLogs(testLogs(1, 150)))
	segCount := len(store1.segments)
	require.NoError(t, store1.Close())

	store2, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	defer store2.Close()

	assert.Equal(t, segCount, len(store2.segments))

	for i := uint64(1); i <= 150; i++ {
		var log Log
		require.NoError(t, store2.GetLog(i, &log), "entry %d missing after reopen", i)
		assert.Equal(t, i, log.Index)
	}
}

func TestFileLogStore_PersistAndReopen_WithCorruption(t *testing.T) {
	dir, cfg := testConfig(t)

	store1, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	require.NoError(t, store1.StoreLogs(testLogs(1, 20)))

	// Corrupt entry 10 before closing.
	seg := store1.findSegment(10)
	require.NotNil(t, seg)
	rec := seg.index[10]
	corruptOff := rec.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	seg.file.ReadAt(b[:], corruptOff)
	b[0] ^= 0xFF
	seg.file.WriteAt(b[:], corruptOff)

	require.NoError(t, store1.Close())

	// Reopen — disentanglement should detect the corruption.
	store2, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	defer store2.Close()

	faulty, err := store2.GetFaultyEntries()
	require.NoError(t, err)
	require.Len(t, faulty, 1)
	assert.Equal(t, uint64(10), faulty[0].Index)
}

func TestFileLogStore_PersistAndReopen_CorruptionInSealedSegment(t *testing.T) {
	dir, cfg := testConfig(t)

	store1, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)

	// Write enough entries to create multiple segments (MaxEntries=64).
	require.NoError(t, store1.StoreLogs(testLogs(1, 130)))
	require.GreaterOrEqual(t, len(store1.segments), 3)

	// Corrupt entry 10 in the first (sealed) segment.
	seg := store1.findSegment(10)
	require.NotNil(t, seg)
	require.NotEqual(t, store1.active, seg, "entry 10 should be in a sealed segment")
	rec := seg.index[10]
	corruptOff := rec.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	seg.file.ReadAt(b[:], corruptOff)
	b[0] ^= 0xFF
	seg.file.WriteAt(b[:], corruptOff)

	require.NoError(t, store1.Close())

	// Reopen — scanSealedSegments should detect the corruption.
	store2, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	defer store2.Close()

	faulty, err := store2.GetFaultyEntries()
	require.NoError(t, err)
	require.Len(t, faulty, 1)
	assert.Equal(t, uint64(10), faulty[0].Index)
	assert.Equal(t, StatusCorrupted, faulty[0].Status)

	// Other entries should still be readable.
	var log Log
	require.NoError(t, store2.GetLog(1, &log))
	require.NoError(t, store2.GetLog(130, &log))
}

func TestFileLogStore_RollbackOnFlushFailure(t *testing.T) {
	dir, cfg := testConfig(t)
	store, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	defer store.Close()

	// Write some initial entries successfully.
	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	first, _ := store.FirstIndex()
	last, _ := store.LastIndex()
	require.Equal(t, uint64(1), first)
	require.Equal(t, uint64(5), last)

	// Record the segment state before the failing write.
	store.mu.RLock()
	segEntryCount := store.active.entryCount
	segDataWritePos := store.active.dataWritePos
	store.mu.RUnlock()

	// Inject a flush failure.
	injectedErr := errors.New("injected flush error")
	store.mu.Lock()
	store.config.testFlushError = injectedErr
	store.mu.Unlock()

	// Attempt to write more entries — should fail.
	err = store.StoreLogs(testLogs(6, 10))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "injected flush error")

	// Clear the fault.
	store.mu.Lock()
	store.config.testFlushError = nil
	store.mu.Unlock()

	// Verify in-memory state was rolled back.
	store.mu.RLock()
	assert.Equal(t, segEntryCount, store.active.entryCount,
		"entryCount should be rolled back")
	assert.Equal(t, segDataWritePos, store.active.dataWritePos,
		"dataWritePos should be rolled back")
	store.mu.RUnlock()

	// The original entries should still be readable.
	for i := uint64(1); i <= 5; i++ {
		var log Log
		require.NoError(t, store.GetLog(i, &log), "original entry %d should survive", i)
	}

	// The failed entries should not be visible.
	last, _ = store.LastIndex()
	assert.Equal(t, uint64(5), last)
	var log Log
	assert.Equal(t, ErrLogNotFound, store.GetLog(6, &log))

	// A subsequent write should succeed and pick up from the correct position.
	require.NoError(t, store.StoreLogs(testLogs(6, 8)))
	last, _ = store.LastIndex()
	assert.Equal(t, uint64(8), last)
	for i := uint64(6); i <= 8; i++ {
		require.NoError(t, store.GetLog(i, &log))
		assert.Equal(t, i, log.Index)
	}
}

// The data write position is derived from the identifiers rather than stored,
// so garbage in the header bytes it used to occupy must not affect recovery.
func TestFileLogStore_DataWritePosNotReadFromHeader(t *testing.T) {
	dir, cfg := testConfig(t)

	store1, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	require.NoError(t, store1.StoreLogs(testLogs(1, 10)))

	seg := store1.active
	expected := seg.dataWritePos
	segPath := seg.path

	// Scribble over the header bytes that previously held dataWritePos.
	var garbage [8]byte
	binary.BigEndian.PutUint64(garbage[:], 0xDEADBEEF)
	_, err = seg.file.WriteAt(garbage[:], 36)
	require.NoError(t, err)
	require.NoError(t, store1.Close())

	store2, err := NewFileLogStore(dir, cfg)
	require.NoError(t, err)
	defer store2.Close()

	require.Len(t, store2.segments, 1)
	assert.Equal(t, segPath, store2.segments[0].path)
	assert.Equal(t, expected, store2.segments[0].dataWritePos,
		"write position must be derived from identifiers, not the header")

	// Appending after recovery must not clobber the last existing entry.
	require.NoError(t, store2.StoreLogs(testLogs(11, 12)))
	for i := uint64(1); i <= 12; i++ {
		var log Log
		require.NoError(t, store2.GetLog(i, &log), "entry %d", i)
		assert.Equal(t, i, log.Index)
	}
}

// Deleting a suffix frees the data those entries occupied, so the write
// position moves back and the next append reuses the space.
func TestFileLogStore_DeleteRange_SuffixReclaimsSpace(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 10)))

	seg := store.active
	posBefore := seg.dataWritePos
	endOfSeven := entryEnd(seg.index[7])
	require.Greater(t, posBefore, endOfSeven)

	require.NoError(t, store.DeleteRange(8, 10))

	assert.Equal(t, endOfSeven, seg.dataWritePos,
		"write position should fall back to the end of entry 7")

	// The next append reuses the reclaimed region.
	require.NoError(t, store.StoreLogs(testLogs(8, 8)))
	assert.Equal(t, endOfSeven, seg.index[8].DataOffset)

	var log Log
	require.NoError(t, store.GetLog(8, &log))
	assert.Equal(t, uint64(8), log.Index)

	// Entries before the deleted range are untouched.
	for i := uint64(1); i <= 7; i++ {
		require.NoError(t, store.GetLog(i, &log), "entry %d", i)
		assert.Equal(t, i, log.Index)
	}
}

// An interior delete must not reclaim the hole, because later entries sit
// beyond it in the data region.
func TestFileLogStore_DeleteRange_InteriorKeepsWritePos(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 10)))

	seg := store.active
	posBefore := seg.dataWritePos

	require.NoError(t, store.DeleteRange(3, 5))

	assert.Equal(t, posBefore, seg.dataWritePos,
		"an interior hole must not move the write position")

	var log Log
	for _, i := range []uint64{1, 2, 6, 7, 8, 9, 10} {
		require.NoError(t, store.GetLog(i, &log), "entry %d", i)
		assert.Equal(t, i, log.Index)
	}
}

func TestFileLogStore_MonotonicLogStore(t *testing.T) {
	store := testStore(t)
	assert.True(t, store.IsMonotonic())
}

func TestFileLogStore_ClosedStore(t *testing.T) {
	store := testStore(t)
	require.NoError(t, store.Close())

	var log Log
	assert.Equal(t, ErrStoreNotOpen, store.GetLog(1, &log))
	assert.Equal(t, ErrStoreNotOpen, store.StoreLogs(testLogs(1, 1)))

	_, err := store.GetLogWithIntegrity(1, &log)
	assert.Equal(t, ErrStoreNotOpen, err)
}

// --------------------------------------------------------------------------
// Identifier region physical separation test
// --------------------------------------------------------------------------

func TestFileLogStore_PhysicalSeparation(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 5)))

	seg := store.segments[0]

	// The data region should start after the identifier region.
	expectedDataOffset := int64(segmentHeaderSize) + int64(seg.maxEntries)*int64(identifierSlotSize)
	assert.Equal(t, expectedDataOffset, seg.dataOffset)

	// First identifier slot is at offset 64, first data entry is at
	// dataOffset — verify they are well separated.
	firstSlotOff := seg.slotOffset(0)
	rec := seg.index[1]

	separation := rec.DataOffset - firstSlotOff
	assert.Greater(t, separation, int64(1024),
		"identifier and data should be physically separated")
}

// --------------------------------------------------------------------------
// Entry frame CRC coverage test
// --------------------------------------------------------------------------

func TestFileLogStore_EntryCRCCoversLengthAndData(t *testing.T) {
	store := testStore(t)

	l := &Log{Index: 1, Term: 1, Type: LogCommand, Data: []byte("crc-test")}
	require.NoError(t, store.StoreLog(l))

	seg := store.segments[0]
	rec := seg.index[1]

	// Read the raw frame.
	frameSize := int64(entryLenSize) + int64(rec.DataLen) + int64(entryCRCSize)
	frame := make([]byte, frameSize)
	_, err := seg.file.ReadAt(frame, rec.DataOffset)
	require.NoError(t, err)

	// Verify CRC covers length + data (first payloadEnd bytes).
	payloadEnd := entryLenSize + int(rec.DataLen)
	storedCRC := binary.BigEndian.Uint32(frame[payloadEnd:])
	computedCRC := crc32.ChecksumIEEE(frame[0:payloadEnd])
	assert.Equal(t, computedCRC, storedCRC)

	// Corrupt the length field and verify CRC fails.
	frame[0] ^= 0xFF
	recomputedCRC := crc32.ChecksumIEEE(frame[0:payloadEnd])
	assert.NotEqual(t, storedCRC, recomputedCRC, "CRC should detect length corruption")
}

// --------------------------------------------------------------------------
// Segment file naming test
// --------------------------------------------------------------------------

func TestFileLogStore_SegmentFileNaming(t *testing.T) {
	store := testStore(t)

	require.NoError(t, store.StoreLogs(testLogs(1, 10)))

	files, err := os.ReadDir(store.dir)
	require.NoError(t, err)

	var segFiles []string
	for _, f := range files {
		if filepath.Ext(f.Name()) == segmentFileExt {
			segFiles = append(segFiles, f.Name())
		}
	}

	require.NotEmpty(t, segFiles)
	assert.Equal(t, "seg-00000000000000000001.seg", segFiles[0])
}

func TestFileLogStore_LargeEntries(t *testing.T) {
	store := testStore(t)

	largeData := make([]byte, 4096)
	for i := range largeData {
		largeData[i] = byte(i % 256)
	}

	l := &Log{Index: 1, Term: 1, Type: LogCommand, Data: largeData}
	require.NoError(t, store.StoreLog(l))

	var got Log
	require.NoError(t, store.GetLog(1, &got))
	assert.Equal(t, largeData, got.Data)
}

// A ⟨term, index⟩ pair uniquely identifies a log entry across the cluster, so
// a replacement that encodes to a different length cannot be the same entry.
// Relocating it would break the invariant that index order matches data offset
// order within a segment, so the repair is rejected instead.
func TestFileLogStore_RepairEntry_DifferentSizeRejected(t *testing.T) {
	store := testStore(t)

	logs := testLogs(1, 5)
	require.NoError(t, store.StoreLogs(logs))

	store.mu.Lock()
	store.faultySet[3] = FaultyEntry{Index: 3, Term: 1, Status: StatusCorrupted}
	store.mu.Unlock()

	seg := store.findSegment(3)
	require.NotNil(t, seg)
	before := seg.index[3]
	writePosBefore := seg.dataWritePos

	repairLog := &Log{
		Index:      3,
		Term:       1,
		Type:       LogCommand,
		Data:       []byte("this is a much longer replacement data for the entry"),
		AppendedAt: logs[2].AppendedAt,
	}
	err := store.RepairEntry(repairLog)
	require.ErrorIs(t, err, ErrRepairMismatch)

	// The entry must stay faulty and nothing on disk may have moved.
	faulty, _ := store.GetFaultyEntries()
	require.Len(t, faulty, 1)
	assert.Equal(t, uint64(3), faulty[0].Index)

	assert.Equal(t, before, seg.index[3], "identifier must be unchanged")
	assert.Equal(t, writePosBefore, seg.dataWritePos, "write position must not advance")
}

// Repairing with a different term is a protocol violation: recovery queries
// peers for a specific ⟨term, index⟩, so the replacement's term must match.
func TestFileLogStore_RepairEntry_TermMismatchRejected(t *testing.T) {
	store := testStore(t)

	logs := testLogs(1, 5)
	require.NoError(t, store.StoreLogs(logs))

	store.mu.Lock()
	store.faultySet[2] = FaultyEntry{Index: 2, Term: 1, Status: StatusCorrupted}
	store.mu.Unlock()

	repairLog := *logs[1]
	repairLog.Term = 7

	err := store.RepairEntry(&repairLog)
	require.ErrorIs(t, err, ErrRepairMismatch)

	faulty, _ := store.GetFaultyEntries()
	require.Len(t, faulty, 1)
	assert.Equal(t, uint64(2), faulty[0].Index)
}

// A repair that is byte-identical to the original overwrites in place, leaving
// the identifier slot and the segment header untouched.
func TestFileLogStore_RepairEntry_InPlaceLeavesMetadataUntouched(t *testing.T) {
	store := testStore(t)

	logs := testLogs(1, 5)
	require.NoError(t, store.StoreLogs(logs))

	seg := store.findSegment(3)
	require.NotNil(t, seg)
	before := seg.index[3]
	writePosBefore := seg.dataWritePos
	entryCountBefore := seg.entryCount

	// Corrupt entry 3, then repair it with the original.
	corruptOff := before.DataOffset + int64(entryLenSize) + 1
	var b [1]byte
	seg.file.ReadAt(b[:], corruptOff)
	b[0] ^= 0xFF
	seg.file.WriteAt(b[:], corruptOff)

	store.mu.Lock()
	store.faultySet[3] = FaultyEntry{Index: 3, Term: 1, Status: StatusCorrupted}
	store.mu.Unlock()

	require.NoError(t, store.RepairEntry(logs[2]))

	assert.Equal(t, before, seg.index[3], "identifier must be unchanged")
	assert.Equal(t, writePosBefore, seg.dataWritePos, "write position must be unchanged")
	assert.Equal(t, entryCountBefore, seg.entryCount, "entry count must be unchanged")

	var got Log
	require.NoError(t, store.GetLog(3, &got))
	assert.Equal(t, logs[2].Data, got.Data)
}
