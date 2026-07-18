package qwenimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/intelligencedev/born/tokenizer"
)

type tokenizerCase struct {
	Text string  `json:"text"`
	IDs  []int32 `json:"ids"`
}

// TestTokenizerParity checks born's HF tokenizer loader against transformers'
// AutoTokenizer output for the Qwen3-VL vocabulary (fixture from
// `harness.py dump-tokenizer-cases`). Token ids must match exactly — a single
// divergent id shifts the whole text conditioning.
func TestTokenizerParity(t *testing.T) {
	tokPath := filepath.Join(modelDir(t), "tokenizer")
	if _, err := os.Stat(filepath.Join(tokPath, "tokenizer.json")); err != nil {
		t.Skipf("tokenizer.json not present: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(fixturesDir(t), "tokenizer-cases.json"))
	if err != nil {
		t.Skipf("tokenizer cases fixture not present: %v", err)
	}
	var cases []tokenizerCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("bad fixture: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("no cases")
	}

	tok, err := tokenizer.LoadFromHuggingFace(tokPath)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	for _, c := range cases {
		name := c.Text
		if len(name) > 24 {
			name = name[:24]
		}
		t.Run(name, func(t *testing.T) {
			got, err := tok.Encode(c.Text)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if len(got) != len(c.IDs) {
				t.Fatalf("token count mismatch: got %d %v want %d %v", len(got), got, len(c.IDs), c.IDs)
			}
			for i := range got {
				if int32(got[i]) != c.IDs[i] {
					t.Fatalf("id[%d] mismatch: got %d want %d (got=%v want=%v)", i, got[i], c.IDs[i], got, c.IDs)
				}
			}
		})
	}
}
