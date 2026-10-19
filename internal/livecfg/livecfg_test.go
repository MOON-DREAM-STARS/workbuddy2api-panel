package livecfg

import (
	"testing"
	"time"
)

func TestEmptyToolTurnRetryModelsAreCopiedAcrossSnapshots(t *testing.T) {
	models := []string{"cn:deepseek-v4.1-flash"}
	holder := New(Snapshot{EmptyToolTurnRetryModels: models})
	models[0] = "mutated-after-store"
	loaded := holder.Load()
	if loaded.EmptyToolTurnRetryModels[0] != "cn:deepseek-v4.1-flash" {
		t.Fatalf("stored models mutated: %v", loaded.EmptyToolTurnRetryModels)
	}
	loaded.EmptyToolTurnRetryModels[0] = "mutated-after-load"
	if got := holder.Load().EmptyToolTurnRetryModels[0]; got != "cn:deepseek-v4.1-flash" {
		t.Fatalf("loaded slice mutated snapshot: %q", got)
	}
}

func TestSnapshotCarriesEmptyToolTurnPolicy(t *testing.T) {
	holder := New(Snapshot{EmptyToolTurnMaxRetries: 1, EmptyToolTurnRetryBudget: 75 * time.Second})
	got := holder.Load()
	if got.EmptyToolTurnMaxRetries != 1 || got.EmptyToolTurnRetryBudget.String() != "1m15s" {
		t.Fatalf("policy=%d/%v", got.EmptyToolTurnMaxRetries, got.EmptyToolTurnRetryBudget)
	}
}
