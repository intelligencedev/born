package qwenimage

import (
	"math/rand/v2"
	"testing"
)

// Benchmark dims mirror one DiT wq projection at 512x512 (seq 1536,
// 6144x6144). The ffn matmuls (6144x16384) scale the same way.
func benchMatmulSetup(seq, in, out int) (x []float32, w halfMat, dst []float32) {
	rng := rand.New(rand.NewPCG(1, 2))
	x = make([]float32, seq*in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	wf := make([]float32, in*out)
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	return x, toHalf(wf), make([]float32, seq*out)
}

func BenchmarkMatmulHalf(b *testing.B) {
	const seq, in, out = 1536, 6144, 6144
	x, w, dst := benchMatmulSetup(seq, in, out)
	flops := 2 * float64(seq) * float64(in) * float64(out)
	b.ResetTimer()
	for b.Loop() {
		matmulHalfInto(dst, x, w, seq, in, out)
	}
	b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
}
