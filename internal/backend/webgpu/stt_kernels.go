//go:build windows || linux || darwin

package webgpu

import (
	"fmt"
	"runtime"

	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
	"github.com/intelligencedev/born/internal/tensor"
)

const powShader = /* wgsl */ `
struct Params {
    count: u32,
    exponent_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> exponent: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.count) {
        return;
    }
    let exponent_index = select(index, 0u, params.exponent_count == 1u);
    output[index] = pow(base[index], exponent[exponent_index]);
}
`

const sliceShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_index = gid.x;
    let count = u32(params[0]);
    let rank = u32(params[1]);
    if (output_index >= count) {
        return;
    }

    var remaining = output_index;
    var input_index = 0i;
    for (var dimension = 0u; dimension < rank; dimension++) {
        let output_stride = u32(params[10u + dimension]);
        let coordinate = remaining / output_stride;
        remaining = remaining % output_stride;
        let input_coordinate = params[18u + dimension] + i32(coordinate) * params[26u + dimension];
        input_index += input_coordinate * params[2u + dimension];
    }
    output[output_index] = input[u32(input_index)];
}
`

const instanceNormalizationShader = /* wgsl */ `
struct Params {
    groups: u32,
    channels: u32,
    spatial: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial_sum: array<f32, 256>;
var<workgroup> partial_square_sum: array<f32, 256>;
var<workgroup> normalization: array<f32, 2>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let group = workgroup_id.x;
    let lane = local_id.x;
    if (group >= params.groups) {
        return;
    }
    let base = group * params.spatial;
    var sum = 0.0;
    var square_sum = 0.0;
    for (var index = lane; index < params.spatial; index += 256u) {
        let value = input[base + index];
        sum += value;
        square_sum += value * value;
    }
    partial_sum[lane] = sum;
    partial_square_sum[lane] = square_sum;
    workgroupBarrier();

    for (var width = 128u; width > 0u; width >>= 1u) {
        if (lane < width) {
            partial_sum[lane] += partial_sum[lane + width];
            partial_square_sum[lane] += partial_square_sum[lane + width];
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        let mean = partial_sum[0] / f32(params.spatial);
        let variance = max(partial_square_sum[0] / f32(params.spatial) - mean * mean, 0.0);
        normalization[0] = mean;
        normalization[1] = inverseSqrt(variance + params.epsilon);
    }
    workgroupBarrier();

    let channel = group % params.channels;
    for (var index = lane; index < params.spatial; index += 256u) {
        output[base + index] = (input[base + index] - normalization[0]) *
            normalization[1] * scale[channel] + bias[channel];
    }
}
`

var bglSlice = []gputypes.BindGroupLayoutEntry{
	bglStorage(0, true), bglStorage(1, false), bglStorage(2, true),
}

var bglInstanceNormalization = []gputypes.BindGroupLayoutEntry{
	bglStorage(0, true), bglStorage(1, true), bglStorage(2, true),
	bglStorage(3, false), bglUniform(4),
}

// Pow computes elementwise float32 exponentiation on WebGPU. The exponent may
// be scalar or have the same shape as the base.
func (b *Backend) Pow(base, exponent *tensor.RawTensor) (*tensor.RawTensor, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if base == nil || exponent == nil || base.DType() != tensor.Float32 || exponent.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: Pow requires float32 base and exponent")
	}
	if exponent.NumElements() != 1 && !base.Shape().Equal(exponent.Shape()) {
		return nil, fmt.Errorf("webgpu: Pow exponent shape %v incompatible with base %v", exponent.Shape(), base.Shape())
	}
	return b.runPowLazy(base, exponent)
}

func (b *Backend) runPowLazy(base, exponent *tensor.RawTensor) (*tensor.RawTensor, error) {
	shader := b.compileShader("pow", powShader)
	entry := b.getOrCreatePipeline("pow", shader, bglBinary)
	baseInput := b.getOrCreateInputBuffer(base)
	exponentInput := b.getOrCreateInputBuffer(exponent)
	transient, lazyDatas := collectSTTInputs(baseInput, exponentInput)

	resultSize := uint64(base.ByteSize()) //nolint:gosec // tensor sizes fit GPU address space
	resultBuffer, err := b.gpuPool.Acquire(resultSize)
	if err != nil {
		return nil, fmt.Errorf("webgpu: Pow result buffer: %w", err)
	}
	params := make([]byte, 16)
	putUint32LE(params[0:4], uint32(base.NumElements()))     //nolint:gosec // tensor size fits u32 dispatch
	putUint32LE(params[4:8], uint32(exponent.NumElements())) //nolint:gosec // tensor size fits u32 dispatch
	paramsBuffer := b.createUniformBuffer(params)
	transient = append(transient, paramsBuffer)
	bindGroup := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(baseInput.buffer, uint64(base.ByteSize())),         //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(exponentInput.buffer, uint64(exponent.ByteSize())), //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(resultBuffer, resultSize),
		bufBinding(paramsBuffer, 16),
	})
	return b.addComputePassToEncoder(
		entry.pipeline, bindGroup, uint32((base.NumElements()+255)/256), 1, 1, //nolint:gosec // bounded tensor size
		resultBuffer, resultSize, base.Shape(), tensor.Float32,
		lazyResources{buffers: transient, lazyDatas: lazyDatas},
	)
}

// Slice extracts an arbitrary positive- or negative-step float32 tensor slice
// without reading the source back to the host.
func (b *Backend) Slice(input *tensor.RawTensor, starts, ends, axes, steps []int64) (*tensor.RawTensor, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if input == nil || input.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: Slice requires float32 input")
	}
	shape, normalizedStarts, normalizedSteps, err := normalizeSlice(input.Shape(), starts, ends, axes, steps)
	if err != nil {
		return nil, err
	}
	if len(shape) > 8 {
		return nil, fmt.Errorf("webgpu: Slice supports rank <= 8, got %d", len(shape))
	}
	if shape.NumElements() == 0 {
		return tensor.Slice(input, starts, ends, axes, steps)
	}
	return b.runSliceLazy(input, shape, normalizedStarts, normalizedSteps)
}

func (b *Backend) runSliceLazy(input *tensor.RawTensor, outputShape tensor.Shape, starts, steps []int) (*tensor.RawTensor, error) {
	shader := b.compileShader("slice", sliceShader)
	entry := b.getOrCreatePipeline("slice", shader, bglSlice)
	inputResult := b.getOrCreateInputBuffer(input)
	transient, lazyDatas := collectSTTInputs(inputResult)

	resultSize := uint64(outputShape.NumElements() * tensor.Float32.Size()) //nolint:gosec // tensor sizes fit GPU address space
	resultBuffer, err := b.gpuPool.Acquire(resultSize)
	if err != nil {
		return nil, fmt.Errorf("webgpu: Slice result buffer: %w", err)
	}
	params := make([]byte, 34*4)
	putUint32LE(params[0:4], uint32(outputShape.NumElements())) //nolint:gosec // bounded tensor size
	putUint32LE(params[4:8], uint32(len(outputShape)))          //nolint:gosec // rank <= 8
	inputStrides := input.Shape().ComputeStrides()
	outputStrides := outputShape.ComputeStrides()
	for dimension := range outputShape {
		putUint32LE(params[(2+dimension)*4:], uint32(inputStrides[dimension]))   //nolint:gosec // bounded tensor stride
		putUint32LE(params[(10+dimension)*4:], uint32(outputStrides[dimension])) //nolint:gosec // bounded tensor stride
		putUint32LE(params[(18+dimension)*4:], uint32(int32(starts[dimension])))
		putUint32LE(params[(26+dimension)*4:], uint32(int32(steps[dimension])))
	}
	paramsBuffer := b.createBuffer(params, gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	transient = append(transient, paramsBuffer)
	bindGroup := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(inputResult.buffer, uint64(input.ByteSize())), //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(resultBuffer, resultSize),
		bufBinding(paramsBuffer, uint64(len(params))),
	})
	return b.addComputePassToEncoder(
		entry.pipeline, bindGroup, uint32((outputShape.NumElements()+255)/256), 1, 1, //nolint:gosec // bounded tensor size
		resultBuffer, resultSize, outputShape, tensor.Float32,
		lazyResources{buffers: transient, lazyDatas: lazyDatas},
	)
}

// InstanceNormalization runs a fused N-C-spatial reduction and affine transform
// on WebGPU, with one workgroup per (batch, channel) segment.
func (b *Backend) InstanceNormalization(input, scale, bias *tensor.RawTensor, epsilon float32) (*tensor.RawTensor, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if input == nil || scale == nil || bias == nil || input.DType() != tensor.Float32 ||
		scale.DType() != tensor.Float32 || bias.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: InstanceNormalization requires float32 tensors")
	}
	shape := input.Shape()
	if len(shape) < 3 || scale.NumElements() != shape[1] || bias.NumElements() != shape[1] {
		return nil, fmt.Errorf("webgpu: InstanceNormalization invalid shapes %v, %v, %v", shape, scale.Shape(), bias.Shape())
	}
	return b.runInstanceNormalizationLazy(input, scale, bias, epsilon)
}

func (b *Backend) runInstanceNormalizationLazy(input, scale, bias *tensor.RawTensor, epsilon float32) (*tensor.RawTensor, error) {
	shape := input.Shape()
	spatial := 1
	for _, dimension := range shape[2:] {
		spatial *= dimension
	}
	groups := shape[0] * shape[1]
	shader := b.compileShader("instanceNormalization", instanceNormalizationShader)
	entry := b.getOrCreatePipeline("instanceNormalization", shader, bglInstanceNormalization)
	inputResult := b.getOrCreateInputBuffer(input)
	scaleResult := b.getOrCreateInputBuffer(scale)
	biasResult := b.getOrCreateInputBuffer(bias)
	transient, lazyDatas := collectSTTInputs(inputResult, scaleResult, biasResult)

	resultSize := uint64(input.ByteSize()) //nolint:gosec // tensor sizes fit GPU address space
	resultBuffer, err := b.gpuPool.Acquire(resultSize)
	if err != nil {
		return nil, fmt.Errorf("webgpu: InstanceNormalization result buffer: %w", err)
	}
	params := make([]byte, 16)
	putUint32LE(params[0:4], uint32(groups))   //nolint:gosec // bounded tensor dimensions
	putUint32LE(params[4:8], uint32(shape[1])) //nolint:gosec // bounded tensor dimensions
	putUint32LE(params[8:12], uint32(spatial)) //nolint:gosec // bounded tensor dimensions
	putFloat32LE(params[12:16], epsilon)
	paramsBuffer := b.createUniformBuffer(params)
	transient = append(transient, paramsBuffer)
	bindGroup := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(inputResult.buffer, uint64(input.ByteSize())), //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(scaleResult.buffer, uint64(scale.ByteSize())), //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(biasResult.buffer, uint64(bias.ByteSize())),   //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(resultBuffer, resultSize),
		bufBinding(paramsBuffer, 16),
	})
	return b.addComputePassToEncoder(
		entry.pipeline, bindGroup, uint32(groups), 1, 1, //nolint:gosec // bounded tensor dimensions
		resultBuffer, resultSize, shape, tensor.Float32,
		lazyResources{buffers: transient, lazyDatas: lazyDatas},
	)
}

func collectSTTInputs(inputs ...inputBufferResult) ([]*wgpu.Buffer, []*tensor.LazyGPUData) {
	var transient []*wgpu.Buffer
	var lazyDatas []*tensor.LazyGPUData
	for _, input := range inputs {
		if !input.cached {
			transient = append(transient, input.buffer)
		} else if input.gpuData != nil {
			lazyDatas = append(lazyDatas, input.gpuData)
		}
	}
	return transient, lazyDatas
}

func normalizeSlice(inputShape tensor.Shape, starts, ends, axes, steps []int64) (tensor.Shape, []int, []int, error) {
	rank := len(inputShape)
	if len(axes) == 0 {
		axes = make([]int64, len(starts))
		for index := range axes {
			axes[index] = int64(index)
		}
	}
	if len(steps) == 0 {
		steps = make([]int64, len(starts))
		for index := range steps {
			steps[index] = 1
		}
	}
	if len(starts) != len(ends) || len(starts) != len(axes) || len(starts) != len(steps) {
		return nil, nil, nil, fmt.Errorf("webgpu: Slice parameter lengths do not match")
	}
	normalizedStarts := make([]int, rank)
	normalizedEnds := make([]int, rank)
	normalizedSteps := make([]int, rank)
	for dimension, size := range inputShape {
		normalizedEnds[dimension] = size
		normalizedSteps[dimension] = 1
	}
	for index, rawAxis := range axes {
		axis := int(rawAxis)
		if axis < 0 {
			axis += rank
		}
		if axis < 0 || axis >= rank {
			return nil, nil, nil, fmt.Errorf("webgpu: Slice axis %d out of range", rawAxis)
		}
		step := int(steps[index])
		if step == 0 {
			return nil, nil, nil, fmt.Errorf("webgpu: Slice step must not be zero")
		}
		start, end, size := int(starts[index]), int(ends[index]), inputShape[axis]
		if start < 0 {
			start += size
		}
		if end < 0 {
			end += size
		}
		if step > 0 {
			start = clampSTTSlice(start, 0, size)
			end = clampSTTSlice(end, 0, size)
		} else {
			start = clampSTTSlice(start, 0, size-1)
			end = clampSTTSlice(end, -1, size-1)
		}
		normalizedStarts[axis], normalizedEnds[axis], normalizedSteps[axis] = start, end, step
	}
	outputShape := make(tensor.Shape, rank)
	for dimension := range inputShape {
		if normalizedSteps[dimension] > 0 {
			outputShape[dimension] = (normalizedEnds[dimension] - normalizedStarts[dimension] + normalizedSteps[dimension] - 1) / normalizedSteps[dimension]
		} else {
			outputShape[dimension] = (normalizedStarts[dimension] - normalizedEnds[dimension] - normalizedSteps[dimension] - 1) / -normalizedSteps[dimension]
		}
		if outputShape[dimension] < 0 {
			outputShape[dimension] = 0
		}
	}
	return outputShape, normalizedStarts, normalizedSteps, nil
}

func clampSTTSlice(value, low, high int) int {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
