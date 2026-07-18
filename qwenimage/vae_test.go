package qwenimage

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/intelligencedev/born/internal/gguf"
)

// TestVAEDecodeParity decodes the seeded latent fixture through the pure-Go
// Wan 2.1 decoder (2D-reduced) and compares against diffusers
// AutoencoderKLWan (fixtures from convert_vae.py). Conv stacks accumulate,
// so the pixel-float gate is 1e-2 with an eyeball PNG saved alongside.
func TestVAEDecodeParity(t *testing.T) {
	if testing.Short() {
		t.Skip("full VAE decode; skipped in -short")
	}
	path := filepath.Join(modelDir(t), "wan-vae-decoder.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("converted VAE gguf not present (run convert_vae.py): %v", err)
	}
	z, zShape := loadFixture(t, "vae.z")      // (1, 16, 64, 64)
	want, wShape := loadFixture(t, "vae.out") // (1, 3, 512, 512)

	f, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conv, err := gguf.NewTensorConverter(f)
	if err != nil {
		t.Fatalf("converter: %v", err)
	}
	defer func() { _ = conv.Close() }()

	v, err := LoadVAE(DefaultVAEConfig(), ggufWeights{conv})
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	start := time.Now()
	got, oh, ow, err := v.Decode(z, zShape[2], zShape[3])
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	t.Logf("decode %dx%d in %s", ow, oh, time.Since(start))
	if oh != wShape[2] || ow != wShape[3] {
		t.Fatalf("output %dx%d, want %dx%d", ow, oh, wShape[3], wShape[2])
	}

	m := maxAbsDiff(t, got, want)
	t.Logf("VAE decode parity maxAbs = %g", m)
	if m > 1e-2 {
		t.Fatalf("divergence maxAbs=%g > 1e-2", m)
	}
}
