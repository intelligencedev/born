package qwenimage

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/intelligencedev/born/internal/gguf"
)

// TestDiTFullParity runs one full 12B Krea 2 Turbo DiT forward (real Q4_K_M
// GGUF weights) over the fox conditioning at 512x512 and compares the
// predicted velocity against the harness fixture (same dequantized weights).
func TestDiTFullParity(t *testing.T) {
	if testing.Short() {
		t.Skip("full 12B forward; skipped in -short")
	}
	path := filepath.Join(modelDir(t), "Krea-2-Turbo-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("DiT gguf not present: %v", err)
	}
	latents, latShape := loadFixture(t, "ditstep.latents") // [1, 1024, 64]
	text, textShape := loadFixture(t, "textenc.hidden")    // [1, 512, 12, 2560]
	maskF, _ := loadFixture(t, "textenc.mask")
	posF, posShape := loadFixture(t, "ditstep.pos")
	tsF, _ := loadFixture(t, "ditstep.t")
	want, _ := loadFixture(t, "ditstep.out")

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
	d, err := LoadDiT(DefaultDiTConfig(), ggufWeights{conv})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	t.Logf("12B weights loaded+dequantized in %s (heap %.1f GB)", time.Since(loadStart), float64(mem.HeapAlloc)/1e9)

	imgSeq := latShape[1]
	txtSeq := textShape[1]
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

	fwdStart := time.Now()
	got, err := d.Forward(latents, imgSeq, text, txtSeq, float64(tsF[0]), pos, valid)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	t.Logf("full 12B forward (seq=%d) in %s", txtSeq+imgSeq, time.Since(fwdStart))

	m := maxAbsDiff(t, got, want)
	t.Logf("full-weight DiT single-step parity maxAbs = %g", m)
	if m > 1e-3 {
		t.Fatalf("divergence maxAbs=%g > 1e-3", m)
	}
}
