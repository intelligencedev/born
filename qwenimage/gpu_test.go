package qwenimage

import (
	"math/rand/v2"
	"testing"
	"time"
)

// TestGPUMatmulMatchesCPU verifies the GPU GEMM path against the CPU kernel
// on a DiT-shaped matmul and reports both timings. Skips when no adapter.
func TestGPUMatmulMatchesCPU(t *testing.T) {
	if getGPU() == nil {
		t.Skipf("no GPU: %v", gpuErr)
	}
	const seq, in, out = 512, 1536, 2048
	rng := rand.New(rand.NewPCG(3, 4))
	x := make([]float32, seq*in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	wf := make([]float32, in*out)
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	h := toHalf(wf)

	cpu := make([]float32, seq*out)
	t0 := time.Now()
	matmulHalfInto(cpu, x, h, seq, in, out)
	cpuDur := time.Since(t0)

	m := newMat(h, in, out)
	if m.gpu == nil {
		t.Fatal("expected GPU-resident mat")
	}
	gpu := make([]float32, seq*out)
	t1 := time.Now()
	m.mul(gpu, x, seq)
	gpuDur := time.Since(t1)

	maxAbs := maxAbsDiff(t, gpu, cpu)
	t.Logf("cpu %s | gpu %s | maxAbs %g", cpuDur, gpuDur, maxAbs)
	// Same f16 weights, different f32 accumulation order: tiny drift only.
	if maxAbs > 1e-2 {
		t.Fatalf("GPU/CPU divergence %g", maxAbs)
	}
}
