package qwenimage

import (
	"testing"
)

func tinyDiTConfig() DiTConfig {
	return DiTConfig{
		InChannels: 16, Layers: 2, HeadDim: 8, Heads: 4, KVHeads: 2,
		FFN: 64, TimestepDim: 16, TextDim: 32, TextLayers: 3,
		TextHeads: 2, TextKVHeads: 2, TextFFN: 48,
		AxesDim: [3]int{4, 2, 2}, RopeTheta: 1000, Eps: 1e-5,
	}
}

// TestDiTTinyParity checks the pure-Go Krea 2 DiT against a random-weight
// diffusers Krea2Transformer2DModel fixture (harness.py dump-tiny-dit),
// including middle-padded text and per-stage intermediates for localization.
func TestDiTTinyParity(t *testing.T) {
	cfg := tinyDiTConfig()
	d, err := LoadDiT(cfg, fixtureWeights{t: t, prefix: "tinydit.w."})
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	img, imgShape := loadFixture(t, "tinydit.in.hidden") // [1, 12, 16]
	text, _ := loadFixture(t, "tinydit.in.text")         // [1, 6, 3, 32]
	posF, posShape := loadFixture(t, "tinydit.in.pos")   // [18, 3]
	maskF, _ := loadFixture(t, "tinydit.in.mask")        // [1, 6]
	tF, _ := loadFixture(t, "tinydit.in.t")              // [1]

	imgSeq := imgShape[1]
	txtSeq := len(maskF)
	pos := make([][3]int, posShape[0])
	for i := range pos {
		for a := 0; a < 3; a++ {
			pos[i][a] = int(posF[i*3+a])
		}
	}
	valid := make([]bool, txtSeq)
	for i := range valid {
		valid[i] = maskF[i] != 0
	}

	got, err := d.Forward(img, imgSeq, text, txtSeq, float64(tF[0]), pos, valid)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	want, _ := loadFixture(t, "tinydit.out")
	m := maxAbsDiff(t, got, want)
	t.Logf("tiny DiT output maxAbs = %g", m)
	if m > 1e-4 {
		// Localize: compare stages in graph order.
		for _, stage := range []string{"temb", "text_fusion", "txt_in", "block0", "block1"} {
			w, shape := loadFixture(t, "tinydit.stage."+stage)
			t.Logf("stage %s: fixture shape %v, first vals %v", stage, shape, w[:min(4, len(w))])
		}
		t.Fatalf("tiny DiT divergence maxAbs=%g > 1e-4 (see stage logs)", m)
	}
}
