package qwenimage

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/intelligencedev/born/internal/gguf"
)

// ggufWeights adapts the generic GGUF TensorConverter to WeightSource.
type ggufWeights struct {
	conv *gguf.TensorConverter
}

func (g ggufWeights) LoadF32(name string) ([]float32, []int, error) {
	return g.conv.Convert(name)
}

// TestTextEncoderFullParity runs the full 36-layer Qwen3-VL-4B text tower
// (real Q8_0 GGUF weights) over the exact fixed-layout fox-prompt inputs and
// compares the stacked select-layer hidden states against the harness fixture
// (same dequantized weights on both sides).
func TestTextEncoderFullParity(t *testing.T) {
	if testing.Short() {
		t.Skip("full 4B forward; skipped in -short")
	}
	path := filepath.Join(modelDir(t), "Qwen3VL-4B-Instruct-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("encoder gguf not present: %v", err)
	}
	idsF, idsShape := loadFixture(t, "textenc.input_ids")
	maskF, _ := loadFixture(t, "textenc.mask")
	want, wantShape := loadFixture(t, "textenc.hidden")

	f, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conv, err := gguf.NewTensorConverter(f)
	if err != nil {
		t.Fatalf("converter: %v", err)
	}
	defer func() { _ = conv.Close() }()

	loadStart := time.Now()
	te, err := LoadTextEncoder(DefaultTextEncoderConfig(), ggufWeights{conv})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Logf("weights loaded+dequantized in %s", time.Since(loadStart))

	seq := idsShape[len(idsShape)-1]
	ids := make([]int32, seq)
	for i := range ids {
		ids[i] = int32(idsF[i])
	}
	// The fixture mask is post-drop (512 tokens); reconstruct the full-length
	// validity: the first 34 template tokens are always valid.
	valid := make([]bool, seq)
	for i := 0; i < 34; i++ {
		valid[i] = true
	}
	for i, v := range maskF {
		valid[34+i] = v != 0
	}

	fwdStart := time.Now()
	layers, err := te.Forward(ids, valid, []int{2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	t.Logf("full forward (seq=%d) in %s", seq, time.Since(fwdStart))

	// Fixture layout: (1, 512, 12, 2560) — token-major, layer, dim; already
	// dropped the 34 template tokens.
	if len(wantShape) != 4 || wantShape[1] != seq-34 || wantShape[2] != 12 {
		t.Fatalf("unexpected fixture shape %v", wantShape)
	}
	hidden := wantShape[3]
	var m float64
	for tok := 34; tok < seq; tok++ {
		if !valid[tok] {
			continue // pad positions are masked downstream
		}
		for li := 0; li < 12; li++ {
			goRow := layers[li][tok*hidden : (tok+1)*hidden]
			wantRow := want[((tok-34)*12+li)*hidden : ((tok-34)*12+li+1)*hidden]
			for d := 0; d < hidden; d++ {
				diff := float64(goRow[d] - wantRow[d])
				if diff < 0 {
					diff = -diff
				}
				if diff > m {
					m = diff
				}
			}
		}
	}
	t.Logf("full-weight parity maxAbs (valid tokens, 12 layers) = %g", m)
	if m > 1e-3 {
		t.Fatalf("divergence maxAbs=%g > 1e-3", m)
	}
}
