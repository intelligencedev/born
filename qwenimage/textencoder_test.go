package qwenimage

import (
	"fmt"
	"testing"
)

// fixtureWeights adapts fixture files to the WeightSource interface using a
// name prefix, e.g. "tinytextenc.w." + GGUF tensor name.
type fixtureWeights struct {
	t      *testing.T
	prefix string
}

func (f fixtureWeights) LoadF32(name string) ([]float32, []int, error) {
	data, shape := loadFixture(f.t, f.prefix+name)
	return data, shape, nil
}

// TestTextEncoderTinyParity runs the pure-Go Qwen3 text tower against a
// random-weight tiny Qwen3VLTextModel fixture (harness.py
// dump-tiny-text-encoder), including middle padding to exercise
// cumulative-valid positions and padding-key masking.
func TestTextEncoderTinyParity(t *testing.T) {
	idsF, _ := loadFixture(t, "tinytextenc.input_ids")
	maskF, _ := loadFixture(t, "tinytextenc.mask")

	// 3 layers: selecting {1,2} keeps clear of HF's final-hidden-state
	// convention (the last hidden_states entry has the final norm applied;
	// the real model selects up to layer 35 of 36 and never hits it).
	cfg := TextEncoderConfig{
		Layers: 3, Hidden: 64, Heads: 4, KVHeads: 2, HeadDim: 16,
		FFN: 128, Vocab: 256, Eps: 1e-6, RopeTheta: 5_000_000,
	}
	te, err := LoadTextEncoder(cfg, fixtureWeights{t: t, prefix: "tinytextenc.w."})
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	ids := make([]int32, len(idsF))
	valid := make([]bool, len(maskF))
	for i := range idsF {
		ids[i] = int32(idsF[i])
		valid[i] = maskF[i] != 0
	}

	got, err := te.Forward(ids, valid, []int{1, 2})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	for slot, layer := range []int{1, 2} {
		want, shape := loadFixture(t, fmt.Sprintf("tinytextenc.hidden.l%d", layer))
		if len(shape) != 3 || shape[0] != 1 || shape[1] != len(ids) || shape[2] != cfg.Hidden {
			t.Fatalf("unexpected fixture shape %v", shape)
		}
		// Padding positions carry garbage on both sides (masked downstream);
		// compare valid tokens only.
		var m float64
		for tok := 0; tok < len(ids); tok++ {
			if !valid[tok] {
				continue
			}
			for d := 0; d < cfg.Hidden; d++ {
				idx := tok*cfg.Hidden + d
				diff := float64(got[slot][idx] - want[idx])
				if diff < 0 {
					diff = -diff
				}
				if diff > m {
					m = diff
				}
			}
		}
		t.Logf("layer %d maxAbs (valid tokens) = %g", layer, m)
		if m > 1e-5 {
			t.Fatalf("layer %d divergence maxAbs=%g > 1e-5", layer, m)
		}
	}
}
