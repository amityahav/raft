// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	hclog "github.com/hashicorp/go-hclog"
	"github.com/hashicorp/go-msgpack/v2/codec"
)

// --------------------------------------------------------------------------
// On-disk format constants
// --------------------------------------------------------------------------

const (
	// segmentMagic identifies a CLSTORE segment file.
	segmentMagic = "CLSTORE\x00"

	// segmentVersion is the current on-disk format version.
	segmentVersion uint32 = 1

	// segmentHeaderSize is the fixed size of the segment file header.
	segmentHeaderSize = 64

	// identifierSlotSize is the size of one identifier record in bytes.
	// Layout: [Term:8][Index:8][DataOffset:4][DataLen:4][Flags:4][CRC32:4]
	identifierSlotSize = 32

	// identifierCRCOffset is where the slot's CRC32 begins; the CRC covers
	// the preceding bytes.
	identifierCRCOffset = 28

	// entryLenSize is the size of the entry length prefix (uint32).
	entryLenSize = 4

	// entryCRCSize is the size of the entry CRC32 suffix.
	entryCRCSize = 4

	// entryOverhead is the per-entry framing overhead in the data region.
	entryOverhead = entryLenSize + entryCRCSize

	// segmentFileExt is the file extension for segment files.
	segmentFileExt = ".seg"

	// maxSegmentSize bounds SegmentSize because entry offsets are stored as
	// uint32 in the identifier slot.
	maxSegmentSize = int64(1) << 32
)

// identifierFlagDeleted marks a slot as a tombstone: the entry that once
// occupied it was deliberately deleted (e.g., by log compaction). A tombstone
// is distinct from an unwritten slot (all zeros) so that recovery can tell a
// deliberate deletion from a crash-truncated tail.
const identifierFlagDeleted uint32 = 1

// --------------------------------------------------------------------------
// Default configuration values
// --------------------------------------------------------------------------

const (
	// DefaultSegmentSize is the default preallocated segment file size (64 MB).
	DefaultSegmentSize = 64 * 1024 * 1024

	// DefaultMaxEntriesPerSegment is the maximum number of log entries per
	// segment. This determines the size of the identifier region:
	//   identifierRegionSize = DefaultMaxEntriesPerSegment * identifierSlotSize
	// With 65536 entries × 32 bytes = 2 MB identifier region.
	DefaultMaxEntriesPerSegment = 65536
)

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

var (
	// ErrSegmentFull is returned when a segment cannot accept more entries.
	ErrSegmentFull = errors.New("segment is full")

	// ErrStoreNotOpen is returned when operating on a closed store.
	ErrStoreNotOpen = errors.New("file log store is not open")

	// ErrEntryNotFaulty is returned by RepairEntry when the target entry
	// is not in the faulty set.
	ErrEntryNotFaulty = errors.New("entry is not marked as faulty")

	// ErrRepairMismatch is returned by RepairEntry when the replacement
	// entry is not byte-identical to the entry it replaces. Because a
	// ⟨term, index⟩ pair uniquely identifies a log entry across the
	// cluster, this indicates the caller supplied the wrong entry or that
	// entry encoding is not deterministic.
	ErrRepairMismatch = errors.New("replacement entry does not match the original")

	// ErrCorruptedEntry is returned by GetLog when an entry fails its
	// CRC check. Callers that need the integrity status instead of an
	// error should use GetLogWithIntegrity.
	ErrCorruptedEntry = errors.New("log entry CRC mismatch: data corrupted")

	// ErrNonMonotonicWrite is returned when a StoreLogs call would place an
	// entry at an index below the active segment's base, which the
	// index-addressed layout cannot represent.
	ErrNonMonotonicWrite = errors.New("log entry index precedes active segment base")
)

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------

// FileLogStoreConfig holds tunable parameters for the FileLogStore.
type FileLogStoreConfig struct {
	// SegmentSize is the preallocated size of each segment file in bytes.
	// Default: 64 MB. Must not exceed 4 GiB.
	SegmentSize int64

	// MaxEntriesPerSegment is the maximum number of log entries stored in
	// a single segment. This determines the identifier region size.
	// Default: 65536.
	MaxEntriesPerSegment uint32

	// NoSync disables fsync calls. Useful for testing but MUST NOT be
	// used in production — it defeats crash-corruption disentanglement.
	NoSync bool

	// Logger is used for operational logging.
	Logger hclog.Logger

	// testFlushError, when non-nil, is returned by flushAndSync instead
	// of performing the real flush. Used only for testing rollback logic.
	testFlushError error
}

// DefaultFileLogStoreConfig returns a configuration with sensible defaults.
func DefaultFileLogStoreConfig() FileLogStoreConfig {
	return FileLogStoreConfig{
		SegmentSize:          DefaultSegmentSize,
		MaxEntriesPerSegment: DefaultMaxEntriesPerSegment,
		Logger:               hclog.NewNullLogger(),
	}
}

// dataRegionOffset returns the byte offset where the data region begins
// in a segment file, computed from the identifier region size.
func (c *FileLogStoreConfig) dataRegionOffset() int64 {
	return int64(segmentHeaderSize) + int64(c.MaxEntriesPerSegment)*int64(identifierSlotSize)
}

// --------------------------------------------------------------------------
// identifierRecord — the 32-byte header slot
// --------------------------------------------------------------------------

// identifierRecord is the in-memory representation of a 32-byte identifier
// slot in the segment header. A slot with a valid CRC is a persist record;
// its Flags distinguish a live entry from a tombstone.
type identifierRecord struct {
	Term       uint64
	Index      uint64
	DataOffset int64
	DataLen    uint32
	Flags      uint32
}

// isDeleted reports whether the slot is a tombstone.
func (r identifierRecord) isDeleted() bool {
	return r.Flags&identifierFlagDeleted != 0
}

// encodeIdentifier serializes an identifierRecord into a 32-byte on-disk
// representation with a CRC32 self-check.
//
// On-disk layout (32 bytes):
//
//	[Term:8][Index:8][DataOffset:4][DataLen:4][Flags:4][CRC32:4]
func encodeIdentifier(rec identifierRecord) [identifierSlotSize]byte {
	var buf [identifierSlotSize]byte
	binary.BigEndian.PutUint64(buf[0:8], rec.Term)
	binary.BigEndian.PutUint64(buf[8:16], rec.Index)
	binary.BigEndian.PutUint32(buf[16:20], uint32(rec.DataOffset))
	binary.BigEndian.PutUint32(buf[20:24], rec.DataLen)
	binary.BigEndian.PutUint32(buf[24:28], rec.Flags)
	crc := crc32.ChecksumIEEE(buf[0:identifierCRCOffset])
	binary.BigEndian.PutUint32(buf[identifierCRCOffset:], crc)
	return buf
}

// decodeIdentifier deserializes a 32-byte on-disk identifier slot.
// Returns the record and whether the CRC is valid. An all-zero slot
// (unwritten) returns ok=false, as does a slot whose CRC does not match.
func decodeIdentifier(buf [identifierSlotSize]byte) (rec identifierRecord, ok bool) {
	allZero := true
	for _, b := range buf {
		if b != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return identifierRecord{}, false
	}

	stored := binary.BigEndian.Uint32(buf[identifierCRCOffset:])
	computed := crc32.ChecksumIEEE(buf[0:identifierCRCOffset])
	if stored != computed {
		return identifierRecord{}, false
	}

	rec.Term = binary.BigEndian.Uint64(buf[0:8])
	rec.Index = binary.BigEndian.Uint64(buf[8:16])
	rec.DataOffset = int64(binary.BigEndian.Uint32(buf[16:20]))
	rec.DataLen = binary.BigEndian.Uint32(buf[20:24])
	rec.Flags = binary.BigEndian.Uint32(buf[24:28])
	return rec, true
}

// --------------------------------------------------------------------------
// Entry encoding / decoding
// --------------------------------------------------------------------------

// encodeLogEntry serializes a Log entry to bytes using msgpack.
func encodeLogEntry(l *Log) ([]byte, error) {
	var buf bytes.Buffer
	h := codec.MsgpackHandle{}
	h.TimeNotBuiltin = true
	enc := codec.NewEncoder(&buf, &h)
	if err := enc.Encode(l); err != nil {
		return nil, fmt.Errorf("encode log entry: %w", err)
	}
	return buf.Bytes(), nil
}

// decodeLogEntry deserializes a msgpack-encoded Log entry.
func decodeLogEntry(data []byte, l *Log) error {
	h := codec.MsgpackHandle{}
	dec := codec.NewDecoder(bytes.NewReader(data), &h)
	if err := dec.Decode(l); err != nil {
		return fmt.Errorf("decode log entry: %w", err)
	}
	return nil
}

// entryChecksum computes the CRC32 of an entry frame's payload region,
// binding the entry's ⟨term, index⟩ into the checksum. This ties the bytes
// in the data region to the identity claimed by the identifier: a frame
// written for one ⟨term, index⟩ cannot pass verification when read as a
// different one, so a misdirected write that lands a valid frame at the
// wrong offset is detected.
//
// payload is the on-disk [entryLen:4][encoded] region (the CRC suffix is
// excluded).
func entryChecksum(term, index uint64, payload []byte) uint32 {
	var prefix [16]byte
	binary.BigEndian.PutUint64(prefix[0:8], term)
	binary.BigEndian.PutUint64(prefix[8:16], index)
	h := crc32.NewIEEE()
	_, _ = h.Write(prefix[:])
	_, _ = h.Write(payload)
	return h.Sum32()
}

// --------------------------------------------------------------------------
// segment — a single on-disk segment file
// --------------------------------------------------------------------------

// segment represents a single preallocated segment file that stores a range
// of log entries starting at baseIndex. The identifier region (header) and
// data region are physically separated. An entry with log index i occupies
// identifier slot (i - baseIndex); this single addressing scheme is used by
// reads, writes, deletes, and recovery alike.
type segment struct {
	file *os.File
	path string

	baseIndex  uint64 // first log index this segment can hold
	maxEntries uint32 // capacity from config
	dataOffset int64  // byte offset where data region starts

	dataWritePos int64 // next append position; derived, never persisted

	// index maps log index → identifierRecord for live entries only.
	index map[uint64]identifierRecord

	// minIndex/maxIndex bound the live entries (0 when empty). Maintained
	// incrementally on write and recomputed on delete.
	minIndex uint64
	maxIndex uint64
}

// slotOffset returns the file offset for the identifier slot at the given
// position within this segment (0-based slot number).
func (s *segment) slotOffset(slot uint32) int64 {
	return int64(segmentHeaderSize) + int64(slot)*int64(identifierSlotSize)
}

// hasCapacityForIndex reports whether index can be placed in this segment,
// i.e., its slot falls within the identifier region.
func (s *segment) hasCapacityForIndex(index uint64) bool {
	return index >= s.baseIndex && index-s.baseIndex < uint64(s.maxEntries)
}

// containsIndex reports whether a live entry exists at index.
func (s *segment) containsIndex(idx uint64) bool {
	_, ok := s.index[idx]
	return ok
}

// noteIndex updates the live bounds after adding idx.
func (s *segment) noteIndex(idx uint64) {
	if s.minIndex == 0 || idx < s.minIndex {
		s.minIndex = idx
	}
	if idx > s.maxIndex {
		s.maxIndex = idx
	}
}

// recomputeBounds recomputes minIndex/maxIndex from the live index. Used
// after deletions, which may remove the current extremes.
func (s *segment) recomputeBounds() {
	s.minIndex, s.maxIndex = 0, 0
	for idx := range s.index {
		s.noteIndex(idx)
	}
}

// entryEnd returns the offset just past the end of rec's frame in the data
// region.
func entryEnd(rec identifierRecord) int64 {
	return rec.DataOffset + int64(entryLenSize) + int64(rec.DataLen) + int64(entryCRCSize)
}

// deriveDataWritePos computes the data region write position from the
// identifiers in the index by taking the maximum end offset. This is
// correct because entries grow the data region sequentially and repairs
// overwrite in place, so the furthest-reaching entry marks the append point.
func (s *segment) deriveDataWritePos() int64 {
	pos := s.dataOffset
	for _, rec := range s.index {
		if end := entryEnd(rec); end > pos {
			pos = end
		}
	}
	return pos
}

// --------------------------------------------------------------------------
// FileLogStore — the main store
// --------------------------------------------------------------------------

// FileLogStore is a file-based LogStore that implements the CLSTORE on-disk
// format from "Protocol-Aware Recovery for Consensus-Based Storage" (FAST'18).
//
// It implements both LogStore and CorruptionAwareLogStore, providing:
//   - Per-entry CRC32 checksums (bound to ⟨term, index⟩) for corruption detection
//   - Physically-separated identifiers (persist records) for
//     crash-corruption disentanglement
//   - Entry-level repair capabilities for distributed recovery
//
// The identifier region is the single durable source of truth: entry
// membership, ordering, and the data write position are all derived from it.
type FileLogStore struct {
	mu     sync.RWMutex
	dir    string
	config FileLogStoreConfig

	segments []*segment // sorted by baseIndex
	active   *segment   // current write target (last element of segments)

	firstIndex uint64 // global first live index (0 if empty)
	lastIndex  uint64 // global last live index (0 if empty)

	faultySet map[uint64]FaultyEntry

	closed bool
	logger hclog.Logger
}

// Compile-time interface assertions.
var _ LogStore = (*FileLogStore)(nil)
var _ CorruptionAwareLogStore = (*FileLogStore)(nil)

// --------------------------------------------------------------------------
// Constructor + recovery
// --------------------------------------------------------------------------

// NewFileLogStore creates or opens a FileLogStore in the given directory.
// On open, it scans existing segment files, rebuilds in-memory indexes,
// and runs crash-corruption disentanglement to populate the faulty set.
func NewFileLogStore(dir string, config FileLogStoreConfig) (*FileLogStore, error) {
	if config.SegmentSize == 0 {
		config.SegmentSize = DefaultSegmentSize
	}
	if config.MaxEntriesPerSegment == 0 {
		config.MaxEntriesPerSegment = DefaultMaxEntriesPerSegment
	}
	if config.Logger == nil {
		config.Logger = hclog.NewNullLogger()
	}
	if config.SegmentSize > maxSegmentSize {
		return nil, fmt.Errorf("segment size %d exceeds max %d", config.SegmentSize, maxSegmentSize)
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create log store dir: %w", err)
	}

	store := &FileLogStore{
		dir:       dir,
		config:    config,
		faultySet: make(map[uint64]FaultyEntry),
		logger:    config.Logger,
	}

	if err := store.loadSegments(); err != nil {
		return nil, fmt.Errorf("load segments: %w", err)
	}

	// Scan all sealed segments for genuine storage corruption (bit rot,
	// misdirected writes). Any segment can be corrupted at any time.
	store.scanSealedSegments()

	// Run crash-corruption disentanglement on the active (last) segment.
	// This is the only segment that can have crash-induced partial writes,
	// because older segments were finalized with an fsync before rotation.
	if store.active != nil {
		if _, _, err := store.DisentangleCrashCorruption(); err != nil {
			return nil, fmt.Errorf("disentangle crash corruption: %w", err)
		}
	}

	store.updateGlobalIndexes()

	store.logger.Info("file log store opened",
		"dir", dir,
		"segments", len(store.segments),
		"firstIndex", store.firstIndex,
		"lastIndex", store.lastIndex,
		"faultyEntries", len(store.faultySet),
	)

	return store, nil
}

// loadSegments discovers and opens all existing segment files in the
// store directory, sorted by base index.
func (s *FileLogStore) loadSegments() error {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return fmt.Errorf("read dir: %w", err)
	}

	var segPaths []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), segmentFileExt) {
			segPaths = append(segPaths, filepath.Join(s.dir, e.Name()))
		}
	}
	sort.Strings(segPaths)

	for _, p := range segPaths {
		seg, err := openSegment(p, s.config.MaxEntriesPerSegment)
		if err != nil {
			return fmt.Errorf("open segment %s: %w", p, err)
		}
		s.segments = append(s.segments, seg)
	}

	if len(s.segments) > 0 {
		s.active = s.segments[len(s.segments)-1]
	}

	return nil
}

// updateGlobalIndexes sets firstIndex and lastIndex from the live entries
// across all segments. firstIndex is the minimum live index (so a deleted
// prefix is correctly excluded) and lastIndex is the maximum.
func (s *FileLogStore) updateGlobalIndexes() {
	s.firstIndex = 0
	s.lastIndex = 0
	for _, seg := range s.segments {
		if len(seg.index) == 0 {
			continue
		}
		if s.firstIndex == 0 || seg.minIndex < s.firstIndex {
			s.firstIndex = seg.minIndex
		}
		if seg.maxIndex > s.lastIndex {
			s.lastIndex = seg.maxIndex
		}
	}
}

// scanSealedSegments verifies the integrity of all sealed (non-active)
// segments. Unlike the active segment, sealed segments cannot have
// crash-induced partial writes (they were fsync'd before rotation), so
// any CRC mismatch is genuine storage corruption.
func (s *FileLogStore) scanSealedSegments() {
	for i, seg := range s.segments {
		if i == len(s.segments)-1 {
			// Skip the active segment — handled by DisentangleCrashCorruption.
			break
		}
		for idx, rec := range seg.index {
			status := s.verifyEntryData(seg, rec)
			if status != StatusOK {
				fe := FaultyEntry{
					Index:  idx,
					Term:   rec.Term,
					Status: status,
				}
				s.faultySet[idx] = fe
				s.logger.Warn("corrupted entry in sealed segment",
					"segment", seg.path,
					"index", idx,
					"term", rec.Term,
					"status", status,
				)
			}
		}
	}
}

// --------------------------------------------------------------------------
// Segment creation / opening
// --------------------------------------------------------------------------

// createSegment creates a new preallocated segment file with the given base
// index. The header is written once and never mutated afterward.
func (s *FileLogStore) createSegment(baseIndex uint64) (*segment, error) {
	name := fmt.Sprintf("seg-%020d%s", baseIndex, segmentFileExt)
	path := filepath.Join(s.dir, name)

	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return nil, fmt.Errorf("create segment file: %w", err)
	}

	// Preallocate to configured size.
	if err := f.Truncate(s.config.SegmentSize); err != nil {
		f.Close()
		os.Remove(path)
		return nil, fmt.Errorf("preallocate segment: %w", err)
	}

	// Write the immutable segment header.
	var hdr [segmentHeaderSize]byte
	copy(hdr[0:8], segmentMagic)
	binary.BigEndian.PutUint32(hdr[8:12], segmentVersion)
	binary.BigEndian.PutUint64(hdr[12:20], baseIndex)
	binary.BigEndian.PutUint32(hdr[20:24], s.config.MaxEntriesPerSegment)
	dataOff := s.config.dataRegionOffset()
	binary.BigEndian.PutUint64(hdr[28:36], uint64(dataOff))

	if _, err := f.WriteAt(hdr[:], 0); err != nil {
		f.Close()
		os.Remove(path)
		return nil, fmt.Errorf("write segment header: %w", err)
	}

	if !s.config.NoSync {
		if err := f.Sync(); err != nil {
			f.Close()
			os.Remove(path)
			return nil, fmt.Errorf("sync new segment: %w", err)
		}
	}

	seg := &segment{
		file:         f,
		path:         path,
		baseIndex:    baseIndex,
		maxEntries:   s.config.MaxEntriesPerSegment,
		dataOffset:   dataOff,
		dataWritePos: dataOff,
		index:        make(map[uint64]identifierRecord),
	}

	return seg, nil
}

// openSegment opens an existing segment file, reads its immutable header,
// and scans the entire identifier region to rebuild the in-memory index.
// Live slots populate the index; tombstones and unwritten slots are skipped.
func openSegment(path string, maxEntries uint32) (*segment, error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open segment file: %w", err)
	}

	var hdr [segmentHeaderSize]byte
	if _, err := f.ReadAt(hdr[:], 0); err != nil {
		f.Close()
		return nil, fmt.Errorf("read segment header: %w", err)
	}

	if string(hdr[0:8]) != segmentMagic {
		f.Close()
		return nil, fmt.Errorf("invalid segment magic: %q", hdr[0:8])
	}

	ver := binary.BigEndian.Uint32(hdr[8:12])
	if ver != segmentVersion {
		f.Close()
		return nil, fmt.Errorf("unsupported segment version %d", ver)
	}

	baseIndex := binary.BigEndian.Uint64(hdr[12:20])
	storedMaxEntries := binary.BigEndian.Uint32(hdr[20:24])
	dataOff := int64(binary.BigEndian.Uint64(hdr[28:36]))

	if storedMaxEntries != maxEntries {
		f.Close()
		return nil, fmt.Errorf("segment maxEntries %d != config %d", storedMaxEntries, maxEntries)
	}

	seg := &segment{
		file:       f,
		path:       path,
		baseIndex:  baseIndex,
		maxEntries: maxEntries,
		dataOffset: dataOff,
		index:      make(map[uint64]identifierRecord),
	}

	// Read the entire identifier region in one pass and index the live slots.
	region := make([]byte, int64(maxEntries)*int64(identifierSlotSize))
	if _, err := f.ReadAt(region, int64(segmentHeaderSize)); err != nil {
		f.Close()
		return nil, fmt.Errorf("read identifier region: %w", err)
	}
	var slotBuf [identifierSlotSize]byte
	for i := uint32(0); i < maxEntries; i++ {
		copy(slotBuf[:], region[int(i)*identifierSlotSize:])
		rec, ok := decodeIdentifier(slotBuf)
		if !ok || rec.isDeleted() {
			continue
		}
		seg.index[rec.Index] = rec
		seg.noteIndex(rec.Index)
	}

	seg.dataWritePos = seg.deriveDataWritePos()

	return seg, nil
}

// --------------------------------------------------------------------------
// LogStore implementation
// --------------------------------------------------------------------------

// FirstIndex returns the first known index in the log, or 0 if empty.
func (s *FileLogStore) FirstIndex() (uint64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.firstIndex, nil
}

// LastIndex returns the last known index in the log, or 0 if empty.
func (s *FileLogStore) LastIndex() (uint64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastIndex, nil
}

// GetLog reads the log entry at the given index. Returns ErrLogNotFound if
// the index is not in range. Returns ErrCorruptedEntry if the entry fails
// its CRC check.
func (s *FileLogStore) GetLog(index uint64, log *Log) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return ErrStoreNotOpen
	}

	status, err := s.getLogInternal(index, log)
	if err != nil {
		return err
	}
	if status == StatusCorrupted {
		return ErrCorruptedEntry
	}
	if status == StatusInaccessible {
		return fmt.Errorf("log entry %d: inaccessible", index)
	}
	return nil
}

// StoreLog stores a single log entry.
func (s *FileLogStore) StoreLog(log *Log) error {
	return s.StoreLogs([]*Log{log})
}

// writtenEntry records an entry placed during a StoreLogs batch, so the
// batch can be undone if the durable flush fails.
type writtenEntry struct {
	seg   *segment
	index uint64
}

// segmentSnapshot captures the mutable state of a pre-existing segment so it
// can be restored if a write batch fails to flush durably.
type segmentSnapshot struct {
	dataWritePos int64
	minIndex     uint64
	maxIndex     uint64
}

// StoreLogs stores multiple log entries, then flushes once. If the flush
// fails, in-memory state is rolled back so the store remains consistent with
// what is durable on disk.
func (s *FileLogStore) StoreLogs(logs []*Log) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStoreNotOpen
	}

	if len(logs) == 0 {
		return nil
	}

	snaps := s.snapshotSegments()
	var written []writtenEntry

	for _, l := range logs {
		if err := s.storeLogEntry(l, &written); err != nil {
			s.rollback(snaps, written)
			return err
		}
	}

	if err := s.flushAndSync(); err != nil {
		s.rollback(snaps, written)
		return fmt.Errorf("flush/sync segments: %w", err)
	}

	s.updateGlobalIndexes()
	return nil
}

// snapshotSegments captures the current mutable state of all existing
// segments.
func (s *FileLogStore) snapshotSegments() map[*segment]segmentSnapshot {
	snaps := make(map[*segment]segmentSnapshot, len(s.segments))
	for _, seg := range s.segments {
		snaps[seg] = segmentSnapshot{
			dataWritePos: seg.dataWritePos,
			minIndex:     seg.minIndex,
			maxIndex:     seg.maxIndex,
		}
	}
	return snaps
}

// rollback undoes a failed write batch: entries written during the batch are
// removed from their segments, segments created during the batch are deleted,
// and surviving segments are restored to their snapshot state.
func (s *FileLogStore) rollback(snaps map[*segment]segmentSnapshot, written []writtenEntry) {
	for _, w := range written {
		delete(w.seg.index, w.index)
	}

	var kept []*segment
	for _, seg := range s.segments {
		snap, ok := snaps[seg]
		if !ok {
			// Segment was created during this batch; discard it.
			seg.file.Close()
			os.Remove(seg.path)
			continue
		}
		seg.dataWritePos = snap.dataWritePos
		seg.minIndex = snap.minIndex
		seg.maxIndex = snap.maxIndex
		kept = append(kept, seg)
	}
	s.segments = kept
	if len(s.segments) > 0 {
		s.active = s.segments[len(s.segments)-1]
	} else {
		s.active = nil
	}
}

// storeLogEntry writes a single log entry to the active segment, rotating
// if necessary. Must be called with s.mu held.
func (s *FileLogStore) storeLogEntry(l *Log, written *[]writtenEntry) error {
	if s.active == nil {
		seg, err := s.createSegment(l.Index)
		if err != nil {
			return fmt.Errorf("create initial segment: %w", err)
		}
		s.segments = append(s.segments, seg)
		s.active = seg
	}

	// An index below the active base cannot be represented; Raft avoids this
	// by deleting a conflicting suffix (which resets the active segment)
	// before re-appending.
	if l.Index < s.active.baseIndex {
		return fmt.Errorf("%w: index %d < base %d", ErrNonMonotonicWrite, l.Index, s.active.baseIndex)
	}

	// Rotate if the entry's slot would fall outside the active segment.
	if !s.active.hasCapacityForIndex(l.Index) {
		if err := s.rotateSegment(l.Index); err != nil {
			return fmt.Errorf("rotate segment: %w", err)
		}
	}

	encoded, err := encodeLogEntry(l)
	if err != nil {
		return fmt.Errorf("encode entry %d: %w", l.Index, err)
	}

	seg := s.active

	needed := int64(entryLenSize + len(encoded) + entryCRCSize)
	if seg.dataWritePos+needed > s.config.SegmentSize {
		// Not enough data-region space — rotate.
		if err := s.rotateSegment(l.Index); err != nil {
			return fmt.Errorf("rotate segment (data full): %w", err)
		}
		seg = s.active
	}

	// Write the data frame: [entryLen:4][encoded:var][CRC32:4], where the CRC
	// is bound to ⟨term, index⟩.
	dataPos := seg.dataWritePos
	payloadEnd := entryLenSize + len(encoded)
	frame := make([]byte, needed)
	binary.BigEndian.PutUint32(frame[0:entryLenSize], uint32(len(encoded)))
	copy(frame[entryLenSize:payloadEnd], encoded)
	crc := entryChecksum(l.Term, l.Index, frame[0:payloadEnd])
	binary.BigEndian.PutUint32(frame[payloadEnd:], crc)

	if _, err := seg.file.WriteAt(frame, dataPos); err != nil {
		return fmt.Errorf("write entry %d data: %w", l.Index, err)
	}

	// Write the identifier to the slot addressed by (index - baseIndex).
	rec := identifierRecord{
		Term:       l.Term,
		Index:      l.Index,
		DataOffset: dataPos,
		DataLen:    uint32(len(encoded)),
	}
	idBuf := encodeIdentifier(rec)
	slot := uint32(l.Index - seg.baseIndex)
	if _, err := seg.file.WriteAt(idBuf[:], seg.slotOffset(slot)); err != nil {
		return fmt.Errorf("write entry %d identifier: %w", l.Index, err)
	}

	seg.index[l.Index] = rec
	seg.noteIndex(l.Index)
	seg.dataWritePos = dataPos + needed
	*written = append(*written, writtenEntry{seg: seg, index: l.Index})

	return nil
}

// rotateSegment fsyncs the current active segment (its header is immutable)
// and creates a new one starting at nextBaseIndex.
func (s *FileLogStore) rotateSegment(nextBaseIndex uint64) error {
	if s.active != nil && !s.config.NoSync {
		if err := s.active.file.Sync(); err != nil {
			return fmt.Errorf("sync rotated segment: %w", err)
		}
	}

	seg, err := s.createSegment(nextBaseIndex)
	if err != nil {
		return err
	}
	s.segments = append(s.segments, seg)
	s.active = seg
	return nil
}

// flushAndSync fsyncs the active segment. The header is immutable, so there
// is nothing else to flush.
func (s *FileLogStore) flushAndSync() error {
	if s.config.testFlushError != nil {
		return s.config.testFlushError
	}
	if s.active == nil || s.config.NoSync {
		return nil
	}
	return s.active.file.Sync()
}

// DeleteRange deletes all log entries in the range [min, max] inclusive.
// Complete segments within the range are removed. Within a partially
// overlapping segment, each deleted entry's slot is overwritten with a
// tombstone so recovery can distinguish a deliberate deletion from a
// crash-truncated tail.
func (s *FileLogStore) DeleteRange(min, max uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStoreNotOpen
	}

	var kept []*segment
	for _, seg := range s.segments {
		if len(seg.index) == 0 {
			kept = append(kept, seg)
			continue
		}

		segFirst := seg.minIndex
		segLast := seg.maxIndex

		if segFirst >= min && segLast <= max {
			// Entire segment falls within the delete range.
			seg.file.Close()
			if err := os.Remove(seg.path); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("remove segment %s: %w", seg.path, err)
			}
			continue
		}

		if segFirst > max || segLast < min {
			// Segment is entirely outside the delete range.
			kept = append(kept, seg)
			continue
		}

		// Partial overlap: tombstone the overlapping live entries.
		delStart := segFirst
		if min > delStart {
			delStart = min
		}
		delEnd := segLast
		if max < delEnd {
			delEnd = max
		}
		for idx := delStart; idx <= delEnd; idx++ {
			rec, ok := seg.index[idx]
			if !ok {
				continue
			}
			tomb := encodeIdentifier(identifierRecord{
				Term:  rec.Term,
				Index: idx,
				Flags: identifierFlagDeleted,
			})
			slot := uint32(idx - seg.baseIndex)
			if _, err := seg.file.WriteAt(tomb[:], seg.slotOffset(slot)); err != nil {
				return fmt.Errorf("tombstone slot for index %d: %w", idx, err)
			}
			delete(seg.index, idx)
			delete(s.faultySet, idx)
		}

		seg.recomputeBounds()
		seg.dataWritePos = seg.deriveDataWritePos()

		if !s.config.NoSync {
			if err := seg.file.Sync(); err != nil {
				return fmt.Errorf("sync segment after delete: %w", err)
			}
		}

		kept = append(kept, seg)
	}

	s.segments = kept
	if len(s.segments) > 0 {
		s.active = s.segments[len(s.segments)-1]
	} else {
		s.active = nil
	}
	s.updateGlobalIndexes()
	return nil
}

// --------------------------------------------------------------------------
// CorruptionAwareLogStore implementation
// --------------------------------------------------------------------------

// GetLogWithIntegrity reads a log entry and reports its integrity status.
// Unlike GetLog, a corrupted entry does NOT cause an error — the status
// is set to StatusCorrupted and the Log's Term and Index fields are
// populated from the physically-separated identifier.
func (s *FileLogStore) GetLogWithIntegrity(index uint64, log *Log) (IntegrityStatus, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return StatusInaccessible, ErrStoreNotOpen
	}

	return s.getLogInternal(index, log)
}

// GetFaultyEntries returns all log entries currently known to be faulty.
func (s *FileLogStore) GetFaultyEntries() ([]FaultyEntry, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]FaultyEntry, 0, len(s.faultySet))
	for _, fe := range s.faultySet {
		result = append(result, fe)
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Index < result[j].Index
	})
	return result, nil
}

// RepairEntry overwrites a faulty log entry with correct data from a peer.
func (s *FileLogStore) RepairEntry(log *Log) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrStoreNotOpen
	}

	_, ok := s.faultySet[log.Index]
	if !ok {
		return ErrEntryNotFaulty
	}

	seg := s.findSegment(log.Index)
	if seg == nil {
		return ErrLogNotFound
	}

	rec, exists := seg.index[log.Index]
	if !exists {
		return ErrLogNotFound
	}

	// A log entry is uniquely identified by its ⟨term, index⟩ pair, so the
	// replacement fetched from a peer must be byte-identical to what was
	// originally written. A mismatch means the caller supplied a different
	// entry, so reject it rather than relocating: an entry that moves within
	// a segment breaks the invariant that index order matches data offset
	// order, which recovery relies on to rebuild the write position.
	if log.Term != rec.Term {
		return fmt.Errorf("%w: entry %d term is %d, replacement has term %d",
			ErrRepairMismatch, log.Index, rec.Term, log.Term)
	}

	encoded, err := encodeLogEntry(log)
	if err != nil {
		return fmt.Errorf("encode repair entry: %w", err)
	}

	if uint32(len(encoded)) != rec.DataLen {
		return fmt.Errorf("%w: entry %d is %d bytes, replacement encodes to %d bytes",
			ErrRepairMismatch, log.Index, rec.DataLen, len(encoded))
	}

	// Overwrite the entry frame in place. The identifier is unchanged, so
	// neither the identifier slot nor anything else needs rewriting.
	payloadEnd := entryLenSize + int(rec.DataLen)
	frame := make([]byte, payloadEnd+entryCRCSize)
	binary.BigEndian.PutUint32(frame[0:entryLenSize], rec.DataLen)
	copy(frame[entryLenSize:payloadEnd], encoded)
	crc := entryChecksum(rec.Term, rec.Index, frame[0:payloadEnd])
	binary.BigEndian.PutUint32(frame[payloadEnd:], crc)

	if _, err := seg.file.WriteAt(frame, rec.DataOffset); err != nil {
		return fmt.Errorf("write repair entry %d: %w", log.Index, err)
	}

	if !s.config.NoSync {
		if err := seg.file.Sync(); err != nil {
			return fmt.Errorf("sync after repair: %w", err)
		}
	}

	delete(s.faultySet, log.Index)
	s.logger.Info("repaired faulty entry", "index", log.Index, "term", log.Term)
	return nil
}

// DisentangleCrashCorruption scans the active segment to separate crash-
// induced partial writes from genuine storage corruption.
//
// It reads the identifier region in slot order. Live entries in the
// contiguous written prefix are verified: a data mismatch there is genuine
// corruption (the identifier is a persist record proving the entry was
// durably written), so the entry is added to the faulty set. Tombstones are
// skipped. The first unwritten slot marks the end of the written region; any
// live identifier found *after* that gap is a crash-truncated orphan (an
// interrupted batch whose writes were reordered) and is discarded along with
// everything from the gap onward, preserving the contiguous-prefix invariant.
func (s *FileLogStore) DisentangleCrashCorruption() (lastSafeIndex uint64, faultyEntries []FaultyEntry, err error) {
	if s.active == nil {
		return 0, nil, nil
	}

	seg := s.active

	region := make([]byte, int64(seg.maxEntries)*int64(identifierSlotSize))
	if _, err := seg.file.ReadAt(region, int64(segmentHeaderSize)); err != nil {
		return 0, nil, fmt.Errorf("read identifier region: %w", err)
	}

	firstAbsent := int64(-1)
	hasOrphans := false

	var slotBuf [identifierSlotSize]byte
	for i := uint32(0); i < seg.maxEntries; i++ {
		copy(slotBuf[:], region[int(i)*identifierSlotSize:])
		rec, ok := decodeIdentifier(slotBuf)

		if !ok {
			if firstAbsent < 0 {
				firstAbsent = int64(i)
			}
			continue
		}

		if firstAbsent >= 0 {
			// A written slot after a gap: crash-truncated orphan.
			hasOrphans = true
			continue
		}

		if rec.isDeleted() {
			continue
		}

		status := s.verifyEntryData(seg, rec)
		if status != StatusOK {
			fe := FaultyEntry{Index: rec.Index, Term: rec.Term, Status: status}
			faultyEntries = append(faultyEntries, fe)
			s.faultySet[rec.Index] = fe
			s.logger.Warn("faulty entry detected",
				"index", rec.Index, "term", rec.Term, "status", status)
		}
	}

	if hasOrphans {
		boundary := uint64(firstAbsent)
		for idx := range seg.index {
			if idx-seg.baseIndex >= boundary {
				delete(seg.index, idx)
				delete(s.faultySet, idx)
			}
		}
		var zeroBuf [identifierSlotSize]byte
		for i := uint32(firstAbsent); i < seg.maxEntries; i++ {
			if _, err := seg.file.WriteAt(zeroBuf[:], seg.slotOffset(i)); err != nil {
				return 0, nil, fmt.Errorf("clear orphan slot %d: %w", i, err)
			}
		}
		seg.recomputeBounds()
		seg.dataWritePos = seg.deriveDataWritePos()
		if !s.config.NoSync {
			if err := seg.file.Sync(); err != nil {
				return 0, nil, fmt.Errorf("sync after disentanglement: %w", err)
			}
		}
		s.logger.Info("discarded crash-truncated entries", "fromSlot", firstAbsent)
	}

	lastSafeIndex = seg.maxIndex
	if lastSafeIndex == 0 && len(s.segments) >= 2 {
		lastSafeIndex = s.segments[len(s.segments)-2].maxIndex
	}

	return lastSafeIndex, faultyEntries, nil
}

// --------------------------------------------------------------------------
// Internal helpers
// --------------------------------------------------------------------------

// findSegment returns the segment holding a live entry at index, or nil.
func (s *FileLogStore) findSegment(index uint64) *segment {
	n := len(s.segments)
	i := sort.Search(n, func(i int) bool {
		return s.segments[i].baseIndex > index
	})
	if i == 0 {
		return nil
	}
	seg := s.segments[i-1]
	if seg.containsIndex(index) {
		return seg
	}
	return nil
}

// getLogInternal reads a log entry and returns its integrity status.
// Must be called with at least s.mu.RLock held.
func (s *FileLogStore) getLogInternal(index uint64, log *Log) (IntegrityStatus, error) {
	if s.firstIndex == 0 || index < s.firstIndex || index > s.lastIndex {
		return StatusOK, ErrLogNotFound
	}

	seg := s.findSegment(index)
	if seg == nil {
		return StatusOK, ErrLogNotFound
	}

	rec, ok := seg.index[index]
	if !ok {
		return StatusOK, ErrLogNotFound
	}

	frameSize := int64(entryLenSize) + int64(rec.DataLen) + int64(entryCRCSize)
	frame := make([]byte, frameSize)
	if _, err := seg.file.ReadAt(frame, rec.DataOffset); err != nil {
		return StatusInaccessible, fmt.Errorf("read entry %d data: %w", index, err)
	}

	payloadEnd := entryLenSize + int(rec.DataLen)
	storedCRC := binary.BigEndian.Uint32(frame[payloadEnd : payloadEnd+entryCRCSize])
	computedCRC := entryChecksum(rec.Term, rec.Index, frame[0:payloadEnd])

	if storedCRC != computedCRC {
		// Populate Term and Index from the identifier.
		log.Term = rec.Term
		log.Index = rec.Index
		return StatusCorrupted, nil
	}

	encoded := frame[entryLenSize:payloadEnd]
	if err := decodeLogEntry(encoded, log); err != nil {
		return StatusInaccessible, fmt.Errorf("decode entry %d: %w", index, err)
	}
	return StatusOK, nil
}

// verifyEntryData checks an entry's data-region CRC without decoding it.
func (s *FileLogStore) verifyEntryData(seg *segment, rec identifierRecord) IntegrityStatus {
	frameSize := int64(entryLenSize) + int64(rec.DataLen) + int64(entryCRCSize)
	frame := make([]byte, frameSize)
	if _, err := seg.file.ReadAt(frame, rec.DataOffset); err != nil {
		return StatusInaccessible
	}

	payloadEnd := entryLenSize + int(rec.DataLen)
	storedCRC := binary.BigEndian.Uint32(frame[payloadEnd : payloadEnd+entryCRCSize])
	computedCRC := entryChecksum(rec.Term, rec.Index, frame[0:payloadEnd])

	if storedCRC != computedCRC {
		return StatusCorrupted
	}
	return StatusOK
}

// Close closes the store and all underlying segment files.
func (s *FileLogStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	var errs []error
	for _, seg := range s.segments {
		if err := seg.file.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	s.closed = true
	s.segments = nil
	s.active = nil

	if len(errs) > 0 {
		return fmt.Errorf("close segments: %v", errs)
	}
	return nil
}

// IsMonotonic marks FileLogStore as a MonotonicLogStore: entries are written
// in increasing index order.
func (s *FileLogStore) IsMonotonic() bool {
	return true
}

// compile-time check for MonotonicLogStore
var _ MonotonicLogStore = (*FileLogStore)(nil)
