//go:build windows || linux || darwin

package webgpu

import (
	"fmt"
	"runtime"

	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
	"github.com/intelligencedev/born/internal/tensor"
)

const conv1DShader = /* wgsl */ `
struct Params {
    batch: u32,
    in_channels: u32,
    input_length: u32,
    out_channels: u32,
    in_channels_per_group: u32,
    kernel_length: u32,
    output_length: u32,
    stride: u32,
    pad_left: i32,
    dilation: u32,
    groups: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_position = gid.x;
    let output_channel = gid.y;
    let batch_index = gid.z;
    if (output_position >= params.output_length ||
        output_channel >= params.out_channels ||
        batch_index >= params.batch) {
        return;
    }

    let out_channels_per_group = params.out_channels / params.groups;
    let group_index = output_channel / out_channels_per_group;
    let first_input_channel = group_index * params.in_channels_per_group;
    let input_base = i32(output_position * params.stride) - params.pad_left;
    var sum = 0.0;
    if (params.has_bias != 0u) {
        sum = bias[output_channel];
    }

    for (var input_channel_in_group = 0u;
         input_channel_in_group < params.in_channels_per_group;
         input_channel_in_group++) {
        let input_channel = first_input_channel + input_channel_in_group;
        for (var kernel_position = 0u;
             kernel_position < params.kernel_length;
             kernel_position++) {
            let input_position = input_base + i32(kernel_position * params.dilation);
            if (input_position >= 0 && input_position < i32(params.input_length)) {
                let input_index = ((batch_index * params.in_channels + input_channel) *
                    params.input_length) + u32(input_position);
                let kernel_index = ((output_channel * params.in_channels_per_group +
                    input_channel_in_group) * params.kernel_length) + kernel_position;
                sum += input[input_index] * kernel[kernel_index];
            }
        }
    }

    let output_index = ((batch_index * params.out_channels + output_channel) *
        params.output_length) + output_position;
    output[output_index] = sum;
}
`

var bglConv1D = []gputypes.BindGroupLayoutEntry{
	bglStorage(0, true), bglStorage(1, true), bglStorage(2, true),
	bglStorage(3, false), bglUniform(4),
}

// Conv1D executes an NCL convolution on WebGPU, including grouped/depthwise
// convolution, dilation, asymmetric padding, and optional channel bias.
func (b *Backend) Conv1D(input, kernel, bias *tensor.RawTensor, stride, padLeft, padRight, dilation, groups int) (*tensor.RawTensor, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	result, err := b.runConv1DLazy(input, kernel, bias, stride, padLeft, padRight, dilation, groups)
	if err != nil {
		return nil, err
	}
	if !b.LazyMode {
		_ = result.Data()
	}
	return result, nil
}

func (b *Backend) runConv1DLazy(input, kernel, bias *tensor.RawTensor, stride, padLeft, padRight, dilation, groups int) (*tensor.RawTensor, error) {
	if input == nil || kernel == nil {
		return nil, fmt.Errorf("webgpu: Conv1D requires input and kernel")
	}
	if input.DType() != tensor.Float32 || kernel.DType() != tensor.Float32 {
		return nil, fmt.Errorf("webgpu: Conv1D supports only float32")
	}
	inputShape, kernelShape := input.Shape(), kernel.Shape()
	if len(inputShape) != 3 || len(kernelShape) != 3 {
		return nil, fmt.Errorf("webgpu: Conv1D requires NCL input and OIK kernel, got %v and %v", inputShape, kernelShape)
	}
	batch, inChannels, inputLength := inputShape[0], inputShape[1], inputShape[2]
	outChannels, inChannelsPerGroup, kernelLength := kernelShape[0], kernelShape[1], kernelShape[2]
	if stride < 1 || dilation < 1 || groups < 1 || padLeft < 0 || padRight < 0 {
		return nil, fmt.Errorf("webgpu: Conv1D invalid stride, dilation, groups, or padding")
	}
	if inChannels%groups != 0 || outChannels%groups != 0 || inChannelsPerGroup != inChannels/groups {
		return nil, fmt.Errorf("webgpu: Conv1D invalid group %d for input %v and kernel %v", groups, inputShape, kernelShape)
	}
	if bias != nil && (bias.DType() != tensor.Float32 || bias.NumElements() != outChannels) {
		return nil, fmt.Errorf("webgpu: Conv1D bias must be float32[%d]", outChannels)
	}

	outputLength := (inputLength+padLeft+padRight-dilation*(kernelLength-1)-1)/stride + 1
	if outputLength <= 0 {
		return nil, fmt.Errorf("webgpu: Conv1D output length must be positive, got %d", outputLength)
	}
	resultShape := tensor.Shape{batch, outChannels, outputLength}
	resultSize := uint64(resultShape.NumElements() * tensor.Float32.Size()) //nolint:gosec // tensor sizes fit GPU address space

	shader := b.compileShader("conv1d", conv1DShader)
	entry := b.getOrCreatePipeline("conv1d", shader, bglConv1D)
	inputBuffer := b.getOrCreateInputBuffer(input)
	kernelBuffer := b.getOrCreateInputBuffer(kernel)

	var transientBuffers []*wgpu.Buffer
	var inputLazyData []*tensor.LazyGPUData
	collectInput := func(value inputBufferResult) {
		if !value.cached {
			transientBuffers = append(transientBuffers, value.buffer)
		} else if value.gpuData != nil {
			inputLazyData = append(inputLazyData, value.gpuData)
		}
	}
	collectInput(inputBuffer)
	collectInput(kernelBuffer)

	hasBias := uint32(0)
	biasSize := uint64(4)
	var biasBuffer inputBufferResult
	if bias != nil {
		hasBias = 1
		biasSize = uint64(bias.ByteSize()) //nolint:gosec // tensor sizes fit GPU address space
		biasBuffer = b.getOrCreateInputBuffer(bias)
		collectInput(biasBuffer)
	} else {
		zeroBias := make([]byte, 4)
		biasBuffer = inputBufferResult{buffer: b.createBuffer(zeroBias, gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)}
		transientBuffers = append(transientBuffers, biasBuffer.buffer)
	}

	resultBuffer, err := b.gpuPool.Acquire(resultSize)
	if err != nil {
		for _, buffer := range transientBuffers {
			buffer.Release()
		}
		return nil, fmt.Errorf("webgpu: Conv1D result buffer: %w", err)
	}

	params := make([]byte, 48)
	putUint32LE(params[0:4], uint32(batch))                //nolint:gosec // validated tensor dimensions
	putUint32LE(params[4:8], uint32(inChannels))           //nolint:gosec // validated tensor dimensions
	putUint32LE(params[8:12], uint32(inputLength))         //nolint:gosec // validated tensor dimensions
	putUint32LE(params[12:16], uint32(outChannels))        //nolint:gosec // validated tensor dimensions
	putUint32LE(params[16:20], uint32(inChannelsPerGroup)) //nolint:gosec // validated tensor dimensions
	putUint32LE(params[20:24], uint32(kernelLength))       //nolint:gosec // validated tensor dimensions
	putUint32LE(params[24:28], uint32(outputLength))       //nolint:gosec // validated tensor dimensions
	putUint32LE(params[28:32], uint32(stride))             //nolint:gosec // validated parameters
	putUint32LE(params[32:36], uint32(padLeft))            //nolint:gosec // WGSL reads identical bits as i32
	putUint32LE(params[36:40], uint32(dilation))           //nolint:gosec // validated parameters
	putUint32LE(params[40:44], uint32(groups))             //nolint:gosec // validated parameters
	putUint32LE(params[44:48], hasBias)
	paramsBuffer := b.createUniformBuffer(params)
	transientBuffers = append(transientBuffers, paramsBuffer)

	bindGroup := b.createBindGroupFromBuffers(entry.layout, []bindGroupBuffer{
		bufBinding(inputBuffer.buffer, uint64(input.ByteSize())),   //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(kernelBuffer.buffer, uint64(kernel.ByteSize())), //nolint:gosec // tensor sizes fit GPU address space
		bufBinding(biasBuffer.buffer, biasSize),
		bufBinding(resultBuffer, resultSize),
		bufBinding(paramsBuffer, uint64(len(params))),
	})

	return b.addComputePassToEncoder(
		entry.pipeline, bindGroup,
		uint32((outputLength+7)/8), uint32((outChannels+7)/8), uint32(batch), //nolint:gosec // validated tensor dimensions
		resultBuffer, resultSize, resultShape, tensor.Float32,
		lazyResources{buffers: transientBuffers, lazyDatas: inputLazyData},
	)
}
