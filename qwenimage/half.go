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

// matmulHalfInto computes dst[seq, out] = x[seq, in] · w[in, out] with f16
// weights, parallel over sequence rows.
func matmulHalfInto(dst, x []float32, w halfMat, seq, in, out int) {
	workers := runtime.GOMAXPROCS(0)
	if workers > seq {
		workers = seq
	}
	if workers <= 1 {
		matmulHalfRows(dst, x, w, 0, seq, in, out)
		return
	}
	var wg sync.WaitGroup
	chunk := (seq + workers - 1) / workers
	for start := 0; start < seq; start += chunk {
		end := start + chunk
		if end > seq {
			end = seq
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			matmulHalfRows(dst, x, w, start, end, in, out)
		}(start, end)
	}
	wg.Wait()
}

func matmulHalfRows(dst, x []float32, w halfMat, from, to, in, out int) {
	for t := from; t < to; t++ {
		row := x[t*in : (t+1)*in]
		o := dst[t*out : (t+1)*out]
		for i := range o {
			o[i] = 0
		}
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
