package qwenimage

import (
	"math"
	"runtime"
	"sync"

	"github.com/intelligencedev/born/internal/gguf"
)

// halfMat stores a weight matrix in float16 bits, row-major [in, out] like
// the f32 mats. Halves resident memory AND matmul memory bandwidth for the
// 12B DiT (the f32 form swap-bound a 64 GB machine); values decode through a
// 64K lookup table.
type halfMat []uint16

// f16Table decodes any float16 bit pattern to float32.
var f16Table = func() []float32 {
	t := make([]float32, 1<<16)
	for i := range t {
		t[i] = gguf.Float16ToFloat32(uint16(i))
	}
	return t
}()

// float32ToFloat16 converts with round-to-nearest-even.
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23&0xff) - 127 + 15
	mant := bits & 0x7fffff
	switch {
	case exp >= 0x1f: // overflow/inf/nan
		if bits&0x7fffffff > 0x7f800000 {
			return sign | 0x7e00 // nan
		}
		return sign | 0x7c00 // inf
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		if mant>>(shift-1)&1 != 0 { // round
			half++
		}
		return sign | half
	default:
		half := sign | uint16(exp)<<10 | uint16(mant>>13)
		if mant&0x1000 != 0 { // round-to-nearest (ties away is fine here)
			half++
		}
		return half
	}
}

func toHalf(w []float32) halfMat {
	h := make(halfMat, len(w))
	for i, v := range w {
		h[i] = float32ToFloat16(v)
	}
	return h
}

// tokenTile bounds how many sequence rows share one pass over the weight
// matrix. Each weight row (out×2 bytes of f16) is then reused tokenTile
// times while L1/L2-hot, cutting weight traffic by the same factor — the
// unblocked form streamed the full matrix once per token and ran
// memory-bound (~44 GFLOPS baseline on M3 Max).
const tokenTile = 64

// matmulHalfInto computes dst[seq, out] = x[seq, in] · w[in, out] with f16
// weights, parallel over token tiles, 4-token register unroll inside.
func matmulHalfInto(dst, x []float32, w halfMat, seq, in, out int) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for start := 0; start < seq; start += tokenTile {
		end := start + tokenTile
		if end > seq {
			end = seq
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(start, end int) {
			defer wg.Done()
			matmulHalfTile(dst, x, w, start, end, in, out)
			<-sem
		}(start, end)
	}
	wg.Wait()
}

func matmulHalfTile(dst, x []float32, w halfMat, from, to, in, out int) {
	for i := from * out; i < to*out; i++ {
		dst[i] = 0
	}
	t := from
	for ; t+4 <= to; t += 4 {
		x0 := x[t*in : (t+1)*in]
		x1 := x[(t+1)*in : (t+2)*in]
		x2 := x[(t+2)*in : (t+3)*in]
		x3 := x[(t+3)*in : (t+4)*in]
		d0 := dst[t*out : (t+1)*out]
		d1 := dst[(t+1)*out : (t+2)*out]
		d2 := dst[(t+2)*out : (t+3)*out]
		d3 := dst[(t+3)*out : (t+4)*out]
		i := 0
		for ; i+4 <= in; i += 4 {
			// 4 input rows × 4 tokens: each dst store amortizes 16 FMAs and
			// each f16 decode serves 4 tokens.
			xa0, xa1, xa2, xa3 := x0[i], x1[i], x2[i], x3[i]
			xb0, xb1, xb2, xb3 := x0[i+1], x1[i+1], x2[i+1], x3[i+1]
			xc0, xc1, xc2, xc3 := x0[i+2], x1[i+2], x2[i+2], x3[i+2]
			xd0, xd1, xd2, xd3 := x0[i+3], x1[i+3], x2[i+3], x3[i+3]
			wa := w[i*out : (i+1)*out]
			wb := w[(i+1)*out : (i+2)*out]
			wc := w[(i+2)*out : (i+3)*out]
			wd := w[(i+3)*out : (i+4)*out]
			for j := 0; j < out; j++ {
				va := f16Table[wa[j]]
				vb := f16Table[wb[j]]
				vc := f16Table[wc[j]]
				vd := f16Table[wd[j]]
				d0[j] += xa0*va + xb0*vb + xc0*vc + xd0*vd
				d1[j] += xa1*va + xb1*vb + xc1*vc + xd1*vd
				d2[j] += xa2*va + xb2*vb + xc2*vc + xd2*vd
				d3[j] += xa3*va + xb3*vb + xc3*vc + xd3*vd
			}
		}
		for ; i < in; i++ {
			xv0, xv1, xv2, xv3 := x0[i], x1[i], x2[i], x3[i]
			wRow := w[i*out : (i+1)*out]
			for j, wbv := range wRow {
				wv := f16Table[wbv]
				d0[j] += xv0 * wv
				d1[j] += xv1 * wv
				d2[j] += xv2 * wv
				d3[j] += xv3 * wv
			}
		}
	}
	for ; t < to; t++ {
		row := x[t*in : (t+1)*in]
		o := dst[t*out : (t+1)*out]
		for i, xv := range row {
			if xv == 0 {
				continue
			}
			wRow := w[i*out : (i+1)*out]
			for j, wb := range wRow {
				o[j] += xv * f16Table[wb]
			}
		}
	}
}

// parallelRows splits [0,rows) across GOMAXPROCS workers.
func parallelRows(rows int, fn func(from, to int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 {
		fn(0, rows)
		return
	}
	var wg sync.WaitGroup
	chunk := (rows + workers - 1) / workers
	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			fn(start, end)
		}(start, end)
	}
	wg.Wait()
}

// matvecHalfInto computes dst[out] = x[in] · w[in, out].
func matvecHalfInto(dst, x []float32, w halfMat, out int) {
	for i := range dst {
		dst[i] = 0
	}
	for i, xv := range x {
		if xv == 0 {
			continue
		}
		wRow := w[i*out : (i+1)*out]
		for j, wb := range wRow {
			dst[j] += xv * f16Table[wb]
		}
	}
}
