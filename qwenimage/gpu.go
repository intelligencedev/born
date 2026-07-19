package qwenimage

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	_ "github.com/gogpu/wgpu/hal/allbackends"
)

// gpuGemmWGSL is the register-blocked f16-weight GEMM validated in
// cmd/gpuprobe: 16x16 threads x (4x4 outputs) = 64x64 output tile per
// workgroup, K staged through workgroup memory in 16-wide chunks, weights
// packed as u32 f16 pairs (unpack2x16float). ~1.67 TFLOPS on M3 Max at
// DiT shapes.
const gpuGemmWGSL = `
struct Params { seq: u32, in_dim: u32, out_dim: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> outbuf: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const BK: u32 = 16u;
const TM: u32 = 4u;
const TN: u32 = 4u;

var<workgroup> xs: array<f32, 1024>;
var<workgroup> ws: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let rowBase = wg.y * 64u + li.y * TM;
    let colBase = wg.x * 64u + li.x * TN;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let ntiles = (p.in_dim + BK - 1u) / BK;
    let lin = li.y * 16u + li.x;
    for (var kt = 0u; kt < ntiles; kt = kt + 1u) {
        let k0 = kt * BK;
        for (var s = 0u; s < 4u; s = s + 1u) {
            let idx = lin * 4u + s;
            let r = idx / BK;
            let k = idx % BK;
            let row = wg.y * 64u + r;
            if (row < p.seq && k0 + k < p.in_dim) {
                xs[idx] = x[row * p.in_dim + k0 + k];
            } else {
                xs[idx] = 0.0;
            }
        }
        for (var s = 0u; s < 2u; s = s + 1u) {
            let idx2 = lin * 2u + s;
            let k = idx2 / 32u;
            let cpair = idx2 % 32u;
            let col = wg.x * 64u + cpair * 2u;
            if (col + 1u < p.out_dim && k0 + k < p.in_dim) {
                let flat = (k0 + k) * p.out_dim + col;
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

// gpuEngine owns the wgpu device, the GEMM pipeline and per-call scratch
// buffers. Weight buffers live on the device; matmuls upload activations,
// dispatch, and read results back (unified memory keeps transfers cheap).
type gpuEngine struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	shader   *wgpu.ShaderModule
	bgl      *wgpu.BindGroupLayout
	pipeline *wgpu.ComputePipeline

	mu      sync.Mutex // wgpu calls serialized per engine
	xBuf    *wgpu.Buffer
	oBuf    *wgpu.Buffer
	sBuf    *wgpu.Buffer
	xCap    uint64
	oCap    uint64
	scratch []byte
}

// gpuAvailable reports whether the GPU path is enabled. Opt out with
// QWENIMAGE_GPU=0.
func gpuAvailable() bool {
	return os.Getenv("QWENIMAGE_GPU") != "0"
}

var (
	gpuOnce sync.Once
	gpuEng  *gpuEngine
	gpuErr  error
)

// getGPU initializes the shared engine once. Returns nil (with a logged
// reason) when no adapter is available; callers fall back to CPU.
func getGPU() *gpuEngine {
	gpuOnce.Do(func() {
		if !gpuAvailable() {
			gpuErr = fmt.Errorf("disabled via QWENIMAGE_GPU=0")
			return
		}
		gpuEng, gpuErr = newGPUEngine()
		if gpuErr != nil {
			fmt.Fprintf(os.Stderr, "qwenimage: GPU unavailable (%v); using CPU\n", gpuErr)
			gpuEng = nil
		}
	})
	return gpuEng
}

func newGPUEngine() (*gpuEngine, error) {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("instance: %w", err)
	}
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("adapter: %w", err)
	}
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("device: %w", err)
	}
	e := &gpuEngine{instance: instance, adapter: adapter, device: device, queue: device.Queue()}

	e.shader, err = device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "qwenimage-gemm", WGSL: gpuGemmWGSL})
	if err != nil {
		return nil, fmt.Errorf("shader: %w", err)
	}
	e.bgl, err = device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "qwenimage-gemm-bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("bgl: %w", err)
	}
	pl, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "qwenimage-gemm-pl", BindGroupLayouts: []*wgpu.BindGroupLayout{e.bgl}})
	if err != nil {
		return nil, fmt.Errorf("pipeline layout: %w", err)
	}
	defer pl.Release()
	e.pipeline, err = device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "qwenimage-gemm", Layout: pl, Module: e.shader, EntryPoint: "main"})
	if err != nil {
		return nil, fmt.Errorf("pipeline: %w", err)
	}
	return e, nil
}

// uploadWeights creates a device-resident buffer holding w (f16 bits,
// row-major [in, out]) and returns it. len(w) must be even (out is even for
// every Krea matrix).
func (e *gpuEngine) uploadWeights(w halfMat) (*wgpu.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "w", Size: uint64(len(w) * 2),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("weight buffer (%d MB): %w", len(w)*2>>20, err)
	}
	bytes := make([]byte, len(w)*2)
	for i, v := range w {
		binary.LittleEndian.PutUint16(bytes[i*2:], v)
	}
	if err := e.queue.WriteBuffer(buf, 0, bytes); err != nil {
		buf.Release()
		return nil, fmt.Errorf("weight upload: %w", err)
	}
	return buf, nil
}

func (e *gpuEngine) ensureScratch(xSize, oSize uint64) error {
	if xSize > e.xCap {
		if e.xBuf != nil {
			e.xBuf.Release()
		}
		var err error
		e.xBuf, err = e.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "x", Size: xSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return err
		}
		e.xCap = xSize
	}
	if oSize > e.oCap {
		if e.oBuf != nil {
			e.oBuf.Release()
		}
		if e.sBuf != nil {
			e.sBuf.Release()
		}
		var err error
		e.oBuf, err = e.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "out", Size: oSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return err
		}
		e.sBuf, err = e.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "staging", Size: oSize, Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
		})
		if err != nil {
			return err
		}
		e.oCap = oSize
	}
	return nil
}

// matmul computes dst[seq, out] = x[seq, in] · W for a device-resident W.
func (e *gpuEngine) matmul(dst, x []float32, wBuf *wgpu.Buffer, wSize uint64, seq, in, out int) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	xSize := uint64(len(x) * 4)
	oSize := uint64(seq*out) * 4
	if err := e.ensureScratch(xSize, oSize); err != nil {
		return err
	}
	if uint64(cap(e.scratch)) < xSize {
		e.scratch = make([]byte, xSize)
	}
	sb := e.scratch[:xSize]
	for i, v := range x {
		binary.LittleEndian.PutUint32(sb[i*4:], math.Float32bits(v))
	}
	if err := e.queue.WriteBuffer(e.xBuf, 0, sb); err != nil {
		return fmt.Errorf("x upload: %w", err)
	}
	params := make([]byte, 16)
	binary.LittleEndian.PutUint32(params[0:], uint32(seq))
	binary.LittleEndian.PutUint32(params[4:], uint32(in))
	binary.LittleEndian.PutUint32(params[8:], uint32(out))
	uBuf, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "params", Size: 16, Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer uBuf.Release()
	if err := e.queue.WriteBuffer(uBuf, 0, params); err != nil {
		return err
	}

	bg, err := e.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label: "gemm-bg", Layout: e.bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: e.xBuf, Size: xSize},
			{Binding: 1, Buffer: wBuf, Size: wSize},
			{Binding: 2, Buffer: e.oBuf, Size: oSize},
			{Binding: 3, Buffer: uBuf, Size: 16},
		},
	})
	if err != nil {
		return fmt.Errorf("bind group: %w", err)
	}
	defer bg.Release()

	enc, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(uint32((out+63)/64), uint32((seq+63)/64), 1)
	if err := pass.End(); err != nil {
		return err
	}
	enc.CopyBufferToBuffer(e.oBuf, 0, e.sBuf, 0, oSize)
	cmd, err := enc.Finish()
	if err != nil {
		return err
	}
	if _, err := e.queue.Submit(cmd); err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := e.sBuf.Map(ctx, wgpu.MapModeRead, 0, oSize); err != nil {
		return fmt.Errorf("map: %w", err)
	}
	rng, err := e.sBuf.MappedRange(0, oSize)
	if err != nil {
		_ = e.sBuf.Unmap()
		return err
	}
	rb := rng.Bytes()
	for i := range dst[:seq*out] {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(rb[i*4:]))
	}
	return e.sBuf.Unmap()
}

// mat is a weight matrix that lives either on the GPU (buffer handle, CPU
// copy released) or on the CPU (halfMat).
type mat struct {
	cpu     halfMat
	gpu     *wgpu.Buffer
	gpuSize uint64
	in, out int
}

// newMat uploads w to the GPU when the shared engine is live (releasing the
// CPU copy), else keeps it on the CPU.
func newMat(w halfMat, in, out int) mat {
	if w == nil {
		return mat{}
	}
	if e := getGPU(); e != nil {
		if buf, err := e.uploadWeights(w); err == nil {
			return mat{gpu: buf, gpuSize: uint64(len(w) * 2), in: in, out: out}
		}
		fmt.Fprintln(os.Stderr, "qwenimage: weight upload failed; keeping CPU copy")
	}
	return mat{cpu: w, in: in, out: out}
}

// mul computes dst[seq, out] = x[seq, in] · W.
func (m *mat) mul(dst, x []float32, seq int) {
	if m.gpu != nil {
		if err := getGPU().matmul(dst, x, m.gpu, m.gpuSize, seq, m.in, m.out); err == nil {
			return
		}
		panic("qwenimage: GPU matmul failed mid-run (no CPU copy retained)")
	}
	matmulHalfInto(dst, x, m.cpu, seq, m.in, m.out)
}

// GPUProbe reports whether the shared GPU engine initializes (for
// diagnostics and tests).
func GPUProbe() string {
	if e := getGPU(); e != nil {
		return "gpu: available"
	}
	return fmt.Sprintf("gpu: unavailable: %v", gpuErr)
}
