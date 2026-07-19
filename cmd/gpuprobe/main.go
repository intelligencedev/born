// Command gpuprobe measures whether gogpu/wgpu's Metal backend can run the
// Krea 2 DiT matmul shape at useful speed. It uploads f16 weights packed as
// u32 pairs (unpack2x16float in-shader; no shader-f16 extension needed),
// runs a 16x16-tiled GEMM, verifies against the CPU kernel, and reports
// GFLOPS.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	_ "github.com/gogpu/wgpu/hal/allbackends"
)

const gemmWGSL = `
struct Params { seq: u32, in_dim: u32, out_dim: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;            // [seq, in]
@group(0) @binding(1) var<storage, read> w: array<u32>;            // [in, out/2] packed 2xf16
@group(0) @binding(2) var<storage, read_write> outbuf: array<f32>; // [seq, out]
@group(0) @binding(3) var<uniform> p: Params;

// Register-blocked GEMM: 16x16 threads per workgroup, each thread computes a
// 4x4 output block -> a 64x64 output tile per workgroup. K advances in
// 16-wide chunks staged through workgroup memory.
const BK: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;

var<workgroup> xs: array<f32, 1024>; // 64 rows x BK
var<workgroup> ws: array<f32, 1024>; // BK x 64 cols

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let rowBase = wg.y * 64u + li.y * TM; // first token this thread writes
    let colBase = wg.x * 64u + li.x * TN; // first output col this thread writes
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let ntiles = (p.in_dim + BK - 1u) / BK;
    let lin = li.y * 16u + li.x; // 0..255
    for (var kt = 0u; kt < ntiles; kt = kt + 1u) {
        let k0 = kt * BK;
        // Stage x tile: 64 rows x 16 k -> 1024 elems, 4 per thread.
        for (var s = 0u; s < 4u; s = s + 1u) {
            let idx = lin * 4u + s;          // 0..1023
            let r = idx / BK;                // 0..63
            let k = idx % BK;
            let row = wg.y * 64u + r;
            if (row < p.seq && k0 + k < p.in_dim) {
                xs[idx] = x[row * p.in_dim + k0 + k];
            } else {
                xs[idx] = 0.0;
            }
        }
        // Stage w tile: 16 k x 64 cols -> 1024 elems, 4 per thread (pairs).
        for (var s = 0u; s < 2u; s = s + 1u) {
            let idx2 = lin * 2u + s;         // 0..511, each covers 2 cols
            let k = idx2 / 32u;              // 0..15
            let cpair = idx2 % 32u;          // 0..31 pairs of columns
            let col = wg.x * 64u + cpair * 2u;
            if (col + 1u < p.out_dim && k0 + k < p.in_dim) {
                let flat = (k0 + k) * p.out_dim + col; // even because out_dim is even
                let pair = unpack2x16float(w[flat >> 1u]);
                ws[k * 64u + cpair * 2u] = pair.x;
                ws[k * 64u + cpair * 2u + 1u] = pair.y;
            } else {
                ws[k * 64u + cpair * 2u] = 0.0;
                ws[k * 64u + cpair * 2u + 1u] = 0.0;
            }
        }
        workgroupBarrier();

        for (var k = 0u; k < BK; k = k + 1u) {
            let xr0 = xs[(li.y * TM + 0u) * BK + k];
            let xr1 = xs[(li.y * TM + 1u) * BK + k];
            let xr2 = xs[(li.y * TM + 2u) * BK + k];
            let xr3 = xs[(li.y * TM + 3u) * BK + k];
            let wc0 = ws[k * 64u + li.x * TN + 0u];
            let wc1 = ws[k * 64u + li.x * TN + 1u];
            let wc2 = ws[k * 64u + li.x * TN + 2u];
            let wc3 = ws[k * 64u + li.x * TN + 3u];
            acc[0]  = acc[0]  + xr0 * wc0;
            acc[1]  = acc[1]  + xr0 * wc1;
            acc[2]  = acc[2]  + xr0 * wc2;
            acc[3]  = acc[3]  + xr0 * wc3;
            acc[4]  = acc[4]  + xr1 * wc0;
            acc[5]  = acc[5]  + xr1 * wc1;
            acc[6]  = acc[6]  + xr1 * wc2;
            acc[7]  = acc[7]  + xr1 * wc3;
            acc[8]  = acc[8]  + xr2 * wc0;
            acc[9]  = acc[9]  + xr2 * wc1;
            acc[10] = acc[10] + xr2 * wc2;
            acc[11] = acc[11] + xr2 * wc3;
            acc[12] = acc[12] + xr3 * wc0;
            acc[13] = acc[13] + xr3 * wc1;
            acc[14] = acc[14] + xr3 * wc2;
            acc[15] = acc[15] + xr3 * wc3;
        }
        workgroupBarrier();
    }

    for (var m = 0u; m < TM; m = m + 1u) {
        let row = rowBase + m;
        if (row >= p.seq) { continue; }
        for (var n = 0u; n < TN; n = n + 1u) {
            let col = colBase + n;
            if (col < p.out_dim) {
                outbuf[row * p.out_dim + col] = acc[m * TN + n];
            }
        }
    }
}
`

func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23&0xff) - 127 + 15
	mant := bits & 0x7fffff
	switch {
	case exp >= 0x1f:
		if bits&0x7fffffff > 0x7f800000 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		if mant>>(shift-1)&1 != 0 {
			half++
		}
		return sign | half
	default:
		half := sign | uint16(exp)<<10 | uint16(mant>>13)
		if mant&0x1000 != 0 {
			half++
		}
		return half
	}
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign << 31
		} else {
			e := uint32(127 - 15 + 1)
			for mant&0x400 == 0 {
				mant <<= 1
				e--
			}
			mant &= 0x3ff
			bits = sign<<31 | e<<23 | mant<<13
		}
	case exp == 0x1f:
		bits = sign<<31 | 0xff<<23 | mant<<13
	default:
		bits = sign<<31 | (exp-15+127)<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

func main() {
	const seq, inDim, outDim = 1536, 6144, 6144
	if err := run(seq, inDim, outDim); err != nil {
		log.Fatalf("FATAL: %v", err)
	}
}

func run(seq, inDim, outDim int) error {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return fmt.Errorf("CreateInstance: %w", err)
	}
	defer instance.Release()
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		return fmt.Errorf("RequestAdapter: %w", err)
	}
	defer adapter.Release()
	info := adapter.Info()
	fmt.Printf("adapter: %+v\n", info)
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		return fmt.Errorf("RequestDevice: %w", err)
	}
	defer device.Release()

	// Host data.
	rng := rand.New(rand.NewPCG(1, 2))
	x := make([]float32, seq*inDim)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	wh := make([]uint16, inDim*outDim)
	for i := range wh {
		wh[i] = float32ToFloat16(float32(rng.NormFloat64()))
	}

	xBytes := make([]byte, len(x)*4)
	for i, v := range x {
		binary.LittleEndian.PutUint32(xBytes[i*4:], math.Float32bits(v))
	}
	wBytes := make([]byte, len(wh)*2)
	for i, v := range wh {
		binary.LittleEndian.PutUint16(wBytes[i*2:], v)
	}
	outSize := uint64(seq * outDim * 4)

	mk := func(label string, size uint64, usage wgpu.BufferUsage) *wgpu.Buffer {
		b, e := device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
		if e != nil {
			panic(fmt.Sprintf("create %s (%d bytes): %v", label, size, e))
		}
		return b
	}
	xBuf := mk("x", uint64(len(xBytes)), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	wBuf := mk("w", uint64(len(wBytes)), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	oBuf := mk("out", outSize, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	sBuf := mk("staging", outSize, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
	uBuf := mk("params", 16, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	defer func() { xBuf.Release(); wBuf.Release(); oBuf.Release(); sBuf.Release(); uBuf.Release() }()

	q := device.Queue()
	if err := q.WriteBuffer(xBuf, 0, xBytes); err != nil {
		return fmt.Errorf("write x: %w", err)
	}
	upStart := time.Now()
	if err := q.WriteBuffer(wBuf, 0, wBytes); err != nil {
		return fmt.Errorf("write w (%d MB): %w", len(wBytes)>>20, err)
	}
	fmt.Printf("weights upload: %d MB in %s\n", len(wBytes)>>20, time.Since(upStart))
	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params[0:], uint32(seq))
	binary.LittleEndian.PutUint32(params[4:], uint32(inDim))
	binary.LittleEndian.PutUint32(params[8:], uint32(outDim))
	if err := q.WriteBuffer(uBuf, 0, params); err != nil {
		return fmt.Errorf("write params: %w", err)
	}

	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "gemm", WGSL: gemmWGSL})
	if err != nil {
		return fmt.Errorf("shader: %w", err)
	}
	defer shader.Release()
	bgl, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "gemm-bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return fmt.Errorf("bgl: %w", err)
	}
	defer bgl.Release()
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label: "gemm-bg", Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: xBuf, Size: uint64(len(xBytes))},
			{Binding: 1, Buffer: wBuf, Size: uint64(len(wBytes))},
			{Binding: 2, Buffer: oBuf, Size: outSize},
			{Binding: 3, Buffer: uBuf, Size: 16},
		},
	})
	if err != nil {
		return fmt.Errorf("bind group: %w", err)
	}
	defer bg.Release()
	pl, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "gemm-pl", BindGroupLayouts: []*wgpu.BindGroupLayout{bgl}})
	if err != nil {
		return fmt.Errorf("pipeline layout: %w", err)
	}
	defer pl.Release()
	pipe, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "gemm", Layout: pl, Module: shader, EntryPoint: "main"})
	if err != nil {
		return fmt.Errorf("pipeline: %w", err)
	}
	defer pipe.Release()

	dispatch := func() error {
		enc, e := device.CreateCommandEncoder(nil)
		if e != nil {
			return e
		}
		pass, e := enc.BeginComputePass(nil)
		if e != nil {
			return e
		}
		pass.SetPipeline(pipe)
		pass.SetBindGroup(0, bg, nil)
		pass.Dispatch(uint32((outDim+63)/64), uint32((seq+63)/64), 1)
		if e := pass.End(); e != nil {
			return e
		}
		cmd, e := enc.Finish()
		if e != nil {
			return e
		}
		_, e = q.Submit(cmd)
		return e
	}
	readback := func() ([]byte, error) {
		enc, e := device.CreateCommandEncoder(nil)
		if e != nil {
			return nil, e
		}
		enc.CopyBufferToBuffer(oBuf, 0, sBuf, 0, outSize)
		cmd, e := enc.Finish()
		if e != nil {
			return nil, e
		}
		if _, e = q.Submit(cmd); e != nil {
			return nil, e
		}
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()
		if e := sBuf.Map(ctx, wgpu.MapModeRead, 0, outSize); e != nil {
			return nil, e
		}
		rg, e := sBuf.MappedRange(0, outSize)
		if e != nil {
			_ = sBuf.Unmap()
			return nil, e
		}
		data := make([]byte, outSize)
		copy(data, rg.Bytes())
		_ = sBuf.Unmap()
		return data, nil
	}

	// Warm-up + correctness.
	if err := dispatch(); err != nil {
		return fmt.Errorf("dispatch: %w", err)
	}
	got, err := readback()
	if err != nil {
		return fmt.Errorf("readback: %w", err)
	}
	// Verify a scattered sample of outputs on the CPU.
	var maxAbs float64
	for s := 0; s < 400; s++ {
		t := (s * 7919) % seq
		j := (s * 104729) % outDim
		var ref float64
		for k := 0; k < inDim; k++ {
			ref += float64(x[t*inDim+k]) * float64(f16ToF32(wh[k*outDim+j]))
		}
		gv := math.Float32frombits(binary.LittleEndian.Uint32(got[(t*outDim+j)*4:]))
		d := math.Abs(float64(gv) - ref)
		if d > maxAbs {
			maxAbs = d
		}
	}
	fmt.Printf("correctness: sampled maxAbs vs CPU f64 ref = %g\n", maxAbs)

	// Timing: N dispatches, one readback to force completion.
	const iters = 10
	start := time.Now()
	for i := 0; i < iters; i++ {
		if err := dispatch(); err != nil {
			return fmt.Errorf("dispatch %d: %w", i, err)
		}
	}
	if _, err := readback(); err != nil {
		return fmt.Errorf("final readback: %w", err)
	}
	el := time.Since(start)
	flops := 2 * float64(seq) * float64(inDim) * float64(outDim) * iters
	fmt.Printf("timing: %d iters in %s -> %.1f GFLOPS\n", iters, el, flops/el.Seconds()/1e9)
	return nil
}
