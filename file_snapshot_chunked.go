// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"

	hclog "github.com/hashicorp/go-hclog"
)

const (
	// DefaultSnapshotChunkSize is the default chunk size used to split a
	// snapshot's state for chunk-level integrity verification and repair.
	// 4096 bytes matches a common filesystem block size, balancing recovery
	// granularity against per-chunk metadata overhead.
	DefaultSnapshotChunkSize = 4096

	// chunkMetaFilePath is the sidecar file holding per-chunk checksums. It
	// is stored alongside — but physically separate from — state.bin, so a
	// misdirected write cannot silently corrupt both the data and the
	// checksums that guard it. This mirrors the identifier/data separation
	// in the CLSTORE log store.
	chunkMetaFilePath = "chunks.meta"
)

// ErrChunkRepairMismatch is returned by RepairChunk when the replacement
// chunk does not match the original chunk's stored checksum. Because CTRL
// assumes snapshots are byte-identical across nodes at a given index, a
// mismatch means the caller supplied the wrong bytes.
var ErrChunkRepairMismatch = errors.New("replacement chunk does not match the original checksum")

// chunkMeta is the on-disk sidecar describing how a snapshot's state.bin is
// chunked and the CRC32 of each chunk. MetaCRC is a self-check over the
// remaining fields so corruption of the sidecar itself is detectable.
type chunkMeta struct {
	ChunkSize int      `json:"chunk_size"`
	Count     int      `json:"count"`
	CRCs      []uint32 `json:"crcs"`
	MetaCRC   uint32   `json:"meta_crc"`
}

// computeMetaCRC computes the self-check CRC over ChunkSize, Count, and CRCs.
func (m *chunkMeta) computeMetaCRC() uint32 {
	h := crc32.NewIEEE()
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], uint32(m.ChunkSize))
	_, _ = h.Write(b[:])
	binary.BigEndian.PutUint32(b[:], uint32(m.Count))
	_, _ = h.Write(b[:])
	for _, c := range m.CRCs {
		binary.BigEndian.PutUint32(b[:], c)
		_, _ = h.Write(b[:])
	}
	return h.Sum32()
}

// chunkLen returns the byte length of chunk i given the total state size.
func (m *chunkMeta) chunkLen(i int, totalSize int64) int {
	start := int64(i) * int64(m.ChunkSize)
	end := start + int64(m.ChunkSize)
	if end > totalSize {
		end = totalSize
	}
	if end < start {
		return 0
	}
	return int(end - start)
}

// --------------------------------------------------------------------------
// chunkHasher — streaming per-chunk CRC32 computation
// --------------------------------------------------------------------------

// chunkHasher accumulates bytes and emits a CRC32 for every chunkSize bytes
// written, plus a final CRC for any trailing partial chunk.
type chunkHasher struct {
	chunkSize int
	cur       hash.Hash32
	curLen    int
	crcs      []uint32
}

func newChunkHasher(chunkSize int) *chunkHasher {
	return &chunkHasher{chunkSize: chunkSize, cur: crc32.NewIEEE()}
}

func (c *chunkHasher) Write(p []byte) {
	for len(p) > 0 {
		space := c.chunkSize - c.curLen
		n := len(p)
		if n > space {
			n = space
		}
		_, _ = c.cur.Write(p[:n])
		c.curLen += n
		p = p[n:]
		if c.curLen == c.chunkSize {
			c.crcs = append(c.crcs, c.cur.Sum32())
			c.cur = crc32.NewIEEE()
			c.curLen = 0
		}
	}
}

// finish flushes any trailing partial chunk and returns the accumulated CRCs.
func (c *chunkHasher) finish() []uint32 {
	if c.curLen > 0 {
		c.crcs = append(c.crcs, c.cur.Sum32())
		c.cur = crc32.NewIEEE()
		c.curLen = 0
	}
	return c.crcs
}

// --------------------------------------------------------------------------
// ChunkedFileSnapshotStore
// --------------------------------------------------------------------------

// ChunkedFileSnapshotStore wraps FileSnapshotStore, adding chunk-level
// integrity verification and repair. It is a drop-in SnapshotStore: existing
// behavior is unchanged, and it additionally satisfies ChunkedSnapshotStore.
//
// Each finalized snapshot gains a chunks.meta sidecar holding a CRC32 per
// chunk of state.bin. Snapshots created by a plain FileSnapshotStore have no
// sidecar; it is generated lazily on first chunk access.
type ChunkedFileSnapshotStore struct {
	*FileSnapshotStore
	chunkSize int
}

// Compile-time interface assertions.
var _ SnapshotStore = (*ChunkedFileSnapshotStore)(nil)
var _ ChunkedSnapshotStore = (*ChunkedFileSnapshotStore)(nil)

// NewChunkedFileSnapshotStoreWithLogger creates a ChunkedFileSnapshotStore.
func NewChunkedFileSnapshotStoreWithLogger(base string, retain int, logger hclog.Logger) (*ChunkedFileSnapshotStore, error) {
	inner, err := NewFileSnapshotStoreWithLogger(base, retain, logger)
	if err != nil {
		return nil, err
	}
	return &ChunkedFileSnapshotStore{
		FileSnapshotStore: inner,
		chunkSize:         DefaultSnapshotChunkSize,
	}, nil
}

// NewChunkedFileSnapshotStore creates a ChunkedFileSnapshotStore writing logs
// to logOutput (defaults to stderr).
func NewChunkedFileSnapshotStore(base string, retain int, logOutput io.Writer) (*ChunkedFileSnapshotStore, error) {
	if logOutput == nil {
		logOutput = os.Stderr
	}
	return NewChunkedFileSnapshotStoreWithLogger(base, retain, hclog.New(&hclog.LoggerOptions{
		Name:   "snapshot",
		Output: logOutput,
		Level:  hclog.DefaultLevel,
	}))
}

// Create begins a new snapshot, returning a sink that computes per-chunk
// checksums as data is written.
func (c *ChunkedFileSnapshotStore) Create(version SnapshotVersion, index, term uint64,
	configuration Configuration, configurationIndex uint64, trans Transport) (SnapshotSink, error) {
	inner, err := c.FileSnapshotStore.Create(version, index, term, configuration, configurationIndex, trans)
	if err != nil {
		return nil, err
	}
	fsink, ok := inner.(*FileSnapshotSink)
	if !ok {
		return nil, fmt.Errorf("unexpected snapshot sink type %T", inner)
	}
	return &ChunkedFileSnapshotSink{
		FileSnapshotSink: fsink,
		store:            c,
		hasher:           newChunkHasher(c.chunkSize),
	}, nil
}

// chunkDir returns the on-disk directory for a finalized snapshot id.
func (c *ChunkedFileSnapshotStore) chunkDir(id string) string {
	return filepath.Join(c.path, id)
}

// loadChunkMeta reads and validates the chunks.meta sidecar for id. If the
// sidecar is missing (a legacy snapshot), it is generated from state.bin.
func (c *ChunkedFileSnapshotStore) loadChunkMeta(id string) (*chunkMeta, error) {
	metaPath := filepath.Join(c.chunkDir(id), chunkMetaFilePath)
	m, err := readChunkMeta(metaPath)
	if err == nil {
		if m.MetaCRC != m.computeMetaCRC() {
			return nil, fmt.Errorf("chunks.meta for snapshot %s is corrupted", id)
		}
		return m, nil
	}
	if !os.IsNotExist(err) {
		return nil, err
	}
	return c.generateChunkMeta(id)
}

// generateChunkMeta builds a chunkMeta for a snapshot lacking a sidecar by
// scanning state.bin, and persists it best-effort.
func (c *ChunkedFileSnapshotStore) generateChunkMeta(id string) (*chunkMeta, error) {
	statePath := filepath.Join(c.chunkDir(id), stateFilePath)
	fh, err := os.Open(statePath)
	if err != nil {
		return nil, fmt.Errorf("open state for chunk meta: %w", err)
	}
	defer func() { _ = fh.Close() }()

	hasher := newChunkHasher(c.chunkSize)
	buf := make([]byte, 64*1024)
	for {
		n, rerr := fh.Read(buf)
		if n > 0 {
			hasher.Write(buf[:n])
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return nil, fmt.Errorf("read state for chunk meta: %w", rerr)
		}
	}

	crcs := hasher.finish()
	m := &chunkMeta{ChunkSize: c.chunkSize, Count: len(crcs), CRCs: crcs}
	m.MetaCRC = m.computeMetaCRC()

	// Persist best-effort so future reads skip regeneration.
	metaPath := filepath.Join(c.chunkDir(id), chunkMetaFilePath)
	if werr := writeChunkMeta(metaPath, m, c.noSync); werr != nil {
		c.logger.Warn("failed to persist generated chunk meta", "id", id, "error", werr)
	}
	return m, nil
}

// stateSize returns the byte size of a snapshot's state.bin.
func (c *ChunkedFileSnapshotStore) stateSize(id string) (int64, error) {
	info, err := os.Stat(filepath.Join(c.chunkDir(id), stateFilePath))
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// ChunkCount returns the number of chunks in the snapshot identified by id.
func (c *ChunkedFileSnapshotStore) ChunkCount(id string) (int, error) {
	m, err := c.loadChunkMeta(id)
	if err != nil {
		return 0, err
	}
	return m.Count, nil
}

// OpenChunk reads a single chunk from the snapshot. The last chunk may be
// shorter than the configured chunk size.
func (c *ChunkedFileSnapshotStore) OpenChunk(id string, chunkIndex int) ([]byte, error) {
	m, err := c.loadChunkMeta(id)
	if err != nil {
		return nil, err
	}
	if chunkIndex < 0 || chunkIndex >= m.Count {
		return nil, fmt.Errorf("chunk index %d out of range [0,%d)", chunkIndex, m.Count)
	}

	statePath := filepath.Join(c.chunkDir(id), stateFilePath)
	fh, err := os.Open(statePath)
	if err != nil {
		return nil, fmt.Errorf("open state file: %w", err)
	}
	defer func() { _ = fh.Close() }()

	off := int64(chunkIndex) * int64(m.ChunkSize)
	buf := make([]byte, m.ChunkSize)
	n, err := fh.ReadAt(buf, off)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("read chunk %d: %w", chunkIndex, err)
	}
	return buf[:n], nil
}

// GetChunkIntegrity verifies a single chunk against its stored checksum.
// An unreadable chunk yields StatusInaccessible; a checksum mismatch yields
// StatusCorrupted. A non-nil error is returned only for problems unrelated to
// data integrity (bad id, out-of-range index, corrupt sidecar).
func (c *ChunkedFileSnapshotStore) GetChunkIntegrity(id string, chunkIndex int) (IntegrityStatus, error) {
	m, err := c.loadChunkMeta(id)
	if err != nil {
		return StatusInaccessible, err
	}
	if chunkIndex < 0 || chunkIndex >= m.Count {
		return StatusInaccessible, fmt.Errorf("chunk index %d out of range [0,%d)", chunkIndex, m.Count)
	}

	data, err := c.OpenChunk(id, chunkIndex)
	if err != nil {
		return StatusInaccessible, nil
	}
	if crc32.ChecksumIEEE(data) != m.CRCs[chunkIndex] {
		return StatusCorrupted, nil
	}
	return StatusOK, nil
}

// RepairChunk overwrites a faulty chunk with correct data from a peer. The
// replacement must be byte-identical to the original (its CRC must match the
// stored checksum), otherwise ErrChunkRepairMismatch is returned. Because the
// bytes are restored exactly, state.bin's whole-file CRC64 remains valid.
func (c *ChunkedFileSnapshotStore) RepairChunk(id string, chunkIndex int, data []byte) error {
	m, err := c.loadChunkMeta(id)
	if err != nil {
		return err
	}
	if chunkIndex < 0 || chunkIndex >= m.Count {
		return fmt.Errorf("chunk index %d out of range [0,%d)", chunkIndex, m.Count)
	}

	size, err := c.stateSize(id)
	if err != nil {
		return fmt.Errorf("stat state file: %w", err)
	}
	if expected := m.chunkLen(chunkIndex, size); len(data) != expected {
		return fmt.Errorf("%w: chunk %d is %d bytes, replacement is %d bytes",
			ErrChunkRepairMismatch, chunkIndex, expected, len(data))
	}
	if crc32.ChecksumIEEE(data) != m.CRCs[chunkIndex] {
		return fmt.Errorf("%w: chunk %d checksum does not match", ErrChunkRepairMismatch, chunkIndex)
	}

	statePath := filepath.Join(c.chunkDir(id), stateFilePath)
	fh, err := os.OpenFile(statePath, os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("open state file for repair: %w", err)
	}
	defer func() { _ = fh.Close() }()

	off := int64(chunkIndex) * int64(m.ChunkSize)
	if _, err := fh.WriteAt(data, off); err != nil {
		return fmt.Errorf("write repaired chunk %d: %w", chunkIndex, err)
	}
	if !c.noSync {
		if err := fh.Sync(); err != nil {
			return fmt.Errorf("sync after repair: %w", err)
		}
	}

	c.logger.Info("repaired snapshot chunk", "id", id, "chunk", chunkIndex)
	return nil
}

// GetFaultyChunks scans every retained snapshot and returns all chunks that
// fail integrity verification.
func (c *ChunkedFileSnapshotStore) GetFaultyChunks() ([]FaultySnapshotChunk, error) {
	snaps, err := c.getSnapshots()
	if err != nil {
		return nil, err
	}

	var faulty []FaultySnapshotChunk
	for _, snap := range snaps {
		m, err := c.loadChunkMeta(snap.ID)
		if err != nil {
			c.logger.Warn("failed to load chunk meta during scan", "id", snap.ID, "error", err)
			continue
		}
		for i := 0; i < m.Count; i++ {
			status, err := c.GetChunkIntegrity(snap.ID, i)
			if err != nil {
				continue
			}
			if status != StatusOK {
				faulty = append(faulty, FaultySnapshotChunk{
					SnapshotID: snap.ID,
					ChunkIndex: i,
					Status:     status,
				})
			}
		}
	}
	return faulty, nil
}

// --------------------------------------------------------------------------
// ChunkedFileSnapshotSink
// --------------------------------------------------------------------------

// ChunkedFileSnapshotSink wraps FileSnapshotSink, tee-ing writes into a
// chunk hasher and writing the chunks.meta sidecar on Close.
type ChunkedFileSnapshotSink struct {
	*FileSnapshotSink
	store  *ChunkedFileSnapshotStore
	hasher *chunkHasher
	closed bool
}

// Write appends to the state file and updates the per-chunk checksums.
func (s *ChunkedFileSnapshotSink) Write(b []byte) (int, error) {
	n, err := s.FileSnapshotSink.Write(b)
	if n > 0 {
		s.hasher.Write(b[:n])
	}
	return n, err
}

// Close finalizes the chunk checksums, writes the chunks.meta sidecar into
// the temporary snapshot directory, then delegates to the inner sink which
// finalizes state.bin and moves the directory (sidecar included) into place.
func (s *ChunkedFileSnapshotSink) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true

	crcs := s.hasher.finish()
	m := &chunkMeta{ChunkSize: s.store.chunkSize, Count: len(crcs), CRCs: crcs}
	m.MetaCRC = m.computeMetaCRC()

	metaPath := filepath.Join(s.dir, chunkMetaFilePath)
	if err := writeChunkMeta(metaPath, m, s.noSync); err != nil {
		_ = s.FileSnapshotSink.Cancel()
		return fmt.Errorf("write chunk meta: %w", err)
	}

	return s.FileSnapshotSink.Close()
}

// --------------------------------------------------------------------------
// chunks.meta serialization
// --------------------------------------------------------------------------

func writeChunkMeta(path string, m *chunkMeta, noSync bool) error {
	fh, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = fh.Close() }()

	buffered := bufio.NewWriter(fh)
	enc := json.NewEncoder(buffered)
	if err := enc.Encode(m); err != nil {
		return err
	}
	if err := buffered.Flush(); err != nil {
		return err
	}
	if !noSync {
		if err := fh.Sync(); err != nil {
			return err
		}
	}
	return nil
}

func readChunkMeta(path string) (*chunkMeta, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = fh.Close() }()

	m := &chunkMeta{}
	dec := json.NewDecoder(bufio.NewReader(fh))
	if err := dec.Decode(m); err != nil {
		return nil, err
	}
	return m, nil
}
