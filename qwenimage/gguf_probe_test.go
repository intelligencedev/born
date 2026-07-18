package qwenimage

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/intelligencedev/born/internal/gguf"
)

// modelDir returns the local Krea 2 model directory, skipping the test when
// the weights are not present (same convention as the supertonic/moonshine
// parity tests).
func modelDir(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home dir: %v", err)
	}
	dir := filepath.Join(home, ".cache", "manifold", "krea2-models")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("krea2 models not present at %s", dir)
	}
	return dir
}

// TestKreaGGUFEnumerate validates that born's generic GGUF parser can open the
// Krea 2 Turbo diffusion GGUF, and logs the tensor inventory that the DiT
// loader will consume.
func TestKreaGGUFEnumerate(t *testing.T) {
	path := filepath.Join(modelDir(t), "Krea-2-Turbo-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("DiT gguf not present: %v", err)
	}
	f, err := gguf.ParseFile(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	t.Logf("architecture=%q name=%q tensors=%d", f.Architecture(), f.Name(), len(f.TensorInfo))
	if len(f.TensorInfo) == 0 {
		t.Fatal("no tensors found")
	}

	byType := map[string]int{}
	names := make([]string, 0, len(f.TensorInfo))
	for i := range f.TensorInfo {
		ti := &f.TensorInfo[i]
		byType[ti.Type.String()]++
		names = append(names, ti.Name)
	}
	t.Logf("quant type histogram: %v", byType)
	sort.Strings(names)
	for _, n := range names[:min(20, len(names))] {
		ti := f.GetTensor(n)
		t.Logf("  %-60s %v %s", n, ti.Dimensions, ti.Type)
	}
}
