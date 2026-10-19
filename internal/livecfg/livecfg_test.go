package livecfg

import "testing"

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
