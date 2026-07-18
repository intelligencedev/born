package qwenimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/intelligencedev/born/internal/gguf"
)

type ggufManifestEntry struct {
	Name  string `json:"name"`
	Shape []int  `json:"shape"`
	Quant string `json:"quant"`
}

// TestGGUFDequantParity compares born's TensorConverter dequantization of the
// real Krea 2 Turbo GGUF against gguf-py reference dequantization of the same
// tensors (fixtures from `harness.py dump-gguf-tensors`). Dequantization is
// deterministic integer math, so the tolerance is essentially exactness.
func TestGGUFDequantParity(t *testing.T) {
	path := filepath.Join(modelDir(t), "Krea-2-Turbo-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("DiT gguf not present: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(fixturesDir(t), "gguf-manifest.json"))
	if err != nil {
		t.Skipf("gguf manifest fixture not present: %v", err)
	}
	var manifest []ggufManifestEntry
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatalf("bad manifest: %v", err)
	}
	if len(manifest) == 0 {
		t.Fatal("empty manifest")
	}

	f, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conv, err := gguf.NewTensorConverter(f)
	if err != nil {
		t.Fatalf("converter: %v", err)
	}
	defer func() { _ = conv.Close() }()

	for _, entry := range manifest {
		t.Run(entry.Name, func(t *testing.T) {
			got, shape, err := conv.Convert(entry.Name)
			if err != nil {
				t.Fatalf("convert: %v", err)
			}
			want, wantShape := loadFixture(t, "gguf."+entry.Name)
			if !equalShape(shape, wantShape) {
				t.Fatalf("shape mismatch: got %v want %v (quant %s)", shape, wantShape, entry.Quant)
			}
			if d := maxAbsDiff(t, got, want); d > 1e-6 {
				t.Fatalf("dequant divergence maxAbs=%g (quant %s)", d, entry.Quant)
			}
		})
	}
}
