// Copyright IBM Corp. 2026
// SPDX-License-Identifier: MPL-2.0

package raft

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCTRL_StoresSatisfyOptionalInterfaces verifies that the CTRL storage
// implementations satisfy both the base interfaces the raft core requires and
// the optional corruption-aware extensions it detects via type assertion.
func TestCTRL_StoresSatisfyOptionalInterfaces(t *testing.T) {
	dir := t.TempDir()

	logs, err := NewFileLogStore(filepath.Join(dir, "logs"), DefaultFileLogStoreConfig())
	require.NoError(t, err)
	t.Cleanup(func() { _ = logs.Close() })

	snaps, err := NewChunkedFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	require.NoError(t, err)

	stable := NewRedundantStableStore(NewInmemStore())

	// Base interfaces (what NewRaft's parameters require).
	var _ LogStore = logs
	var _ StableStore = stable
	var _ SnapshotStore = snaps

	// The raft core detects CTRL capability by asserting to these optional
	// interfaces; both must succeed for the corruption-aware paths to engage.
	var logIface LogStore = logs
	_, ok := logIface.(CorruptionAwareLogStore)
	assert.True(t, ok, "FileLogStore must be detected as CorruptionAwareLogStore")

	var snapIface SnapshotStore = snaps
	_, ok = snapIface.(ChunkedSnapshotStore)
	assert.True(t, ok, "ChunkedFileSnapshotStore must be detected as ChunkedSnapshotStore")
}

// TestCTRL_NewRaftBootsWithCorruptionAwareStores verifies that Raft boots as a
// drop-in with the CTRL stores in place of the standard implementations, with
// no change to core behavior.
func TestCTRL_NewRaftBootsWithCorruptionAwareStores(t *testing.T) {
	dir := t.TempDir()

	logs, err := NewFileLogStore(filepath.Join(dir, "logs"), DefaultFileLogStoreConfig())
	require.NoError(t, err)
	t.Cleanup(func() { _ = logs.Close() })

	snaps, err := NewChunkedFileSnapshotStoreWithLogger(dir, 3, newTestLogger(t))
	require.NoError(t, err)

	stable := NewRedundantStableStore(NewInmemStore())

	_, trans := NewInmemTransport(NewInmemAddr())

	conf := DefaultConfig()
	conf.LocalID = "ctrl-node"
	conf.Logger = newTestLogger(t)

	// Bootstrap a single-node cluster so the node has a configuration.
	cfg := Configuration{Servers: []Server{{
		Suffrage: Voter,
		ID:       conf.LocalID,
		Address:  trans.LocalAddr(),
	}}}
	require.NoError(t, BootstrapCluster(conf, logs, stable, snaps, trans, cfg))

	r, err := NewRaft(conf, &MockFSM{}, logs, stable, snaps, trans)
	require.NoError(t, err)
	require.NotNil(t, r)

	// Boot must succeed and the node must reach a stable running state.
	require.Eventually(t, func() bool {
		return r.State() == Leader
	}, 5*time.Second, 10*time.Millisecond, "single node should elect itself leader")

	require.NoError(t, r.Shutdown().Error())
}
