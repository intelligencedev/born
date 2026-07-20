package moonshine

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/intelligencedev/born/backend/cpu"
)

func TestBackendNameAndClose(t *testing.T) {
	releaseCalls := 0
	stt := &STT{backend: cpu.New(), release: func() { releaseCalls++ }}
	if got := stt.BackendName(); got != "CPU" {
		t.Fatalf("BackendName() = %q, want CPU", got)
	}
	stt.Close()
	stt.Close()
	if releaseCalls != 1 {
		t.Fatalf("release called %d times, want 1", releaseCalls)
	}
}

// Env-gated: MOONSHINE_MODEL_DIR (onnx/ + tokenizer.json), MOONSHINE_REF
// (JSON with audio floats + onnxruntime greedy ids + text).
func TestTranscribeMatchesReference(t *testing.T) {
	md, ref := os.Getenv("MOONSHINE_MODEL_DIR"), os.Getenv("MOONSHINE_REF")
	if md == "" || ref == "" {
		t.Skip("set MOONSHINE_MODEL_DIR and MOONSHINE_REF")
	}
	raw, err := os.ReadFile(ref)
	if err != nil {
		t.Fatal(err)
	}
	var r struct {
		Audio []float32 `json:"audio"`
		IDs   []int64   `json:"ids"`
		Text  string    `json:"text"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	s, err := New(md)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	ids, err := s.TranscribeTokens(r.Audio)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != len(r.IDs) {
		t.Fatalf("ids len %d want %d\n got %v\nwant %v", len(ids), len(r.IDs), ids, r.IDs)
	}
	for i := range r.IDs {
		if ids[i] != r.IDs[i] {
			t.Fatalf("ids[%d]=%d want %d (%v vs %v)", i, ids[i], r.IDs[i], ids, r.IDs)
		}
	}
	if got := s.DecodeTokens(ids); got != r.Text {
		t.Fatalf("text %q want %q", got, r.Text)
	}
	t.Logf("transcribed: %q", r.Text)
}
