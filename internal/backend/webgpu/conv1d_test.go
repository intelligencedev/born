//go:build windows || linux || darwin

package webgpu

import (
	"math"
	"testing"

	"github.com/intelligencedev/born/internal/tensor"
)

func TestConv1DGPU(t *testing.T) {
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}
	backend, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer backend.Release()

	tests := []struct {
		name                                string
		inputShape, kernelShape             tensor.Shape
		input, kernel, bias                 []float32
		stride, padLeft, padRight, dilation int
		groups                              int
	}{
		{
			name:        "bias and asymmetric padding",
			inputShape:  tensor.Shape{1, 2, 4},
			kernelShape: tensor.Shape{2, 2, 3},
			input:       []float32{1, 2, 3, 4, 5, 6, 7, 8},
			kernel:      []float32{1, 0, -1, 0.5, 0, -0.5, -1, 0, 1, -0.5, 0, 0.5},
			bias:        []float32{0.25, -0.5},
			stride:      1,
			padLeft:     2,
			padRight:    1,
			dilation:    1,
			groups:      1,
		},
		{
			name:        "depthwise dilation",
			inputShape:  tensor.Shape{1, 2, 5},
			kernelShape: tensor.Shape{2, 1, 3},
			input:       []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			kernel:      []float32{1, 2, 3, -1, 0.5, 2},
			stride:      1,
			padLeft:     2,
			padRight:    2,
			dilation:    2,
			groups:      2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := newFloat32Tensor(t, test.inputShape, test.input)
			kernel := newFloat32Tensor(t, test.kernelShape, test.kernel)
			var bias *tensor.RawTensor
			if test.bias != nil {
				bias = newFloat32Tensor(t, tensor.Shape{len(test.bias)}, test.bias)
			}

			gotTensor, err := backend.Conv1D(input, kernel, bias, test.stride, test.padLeft, test.padRight, test.dilation, test.groups)
			if err != nil {
				t.Fatalf("Conv1D: %v", err)
			}
			got := gotTensor.AsFloat32()
			want := referenceConv1D(test.input, test.kernel, test.bias, test.inputShape, test.kernelShape,
				test.stride, test.padLeft, test.padRight, test.dilation, test.groups)
			if len(got) != len(want) {
				t.Fatalf("result length = %d, want %d", len(got), len(want))
			}
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 1e-5 {
					t.Fatalf("result[%d] = %v, want %v; got %v", i, got[i], want[i], got)
				}
			}
		})
	}
}

func TestBatchMatMulBroadcastGPU(t *testing.T) {
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}
	backend, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer backend.Release()

	tests := []struct {
		name       string
		aShape     tensor.Shape
		a          []float32
		bShape     tensor.Shape
		b          []float32
		wantShape  tensor.Shape
		wantValues []float32
	}{
		{
			name:       "shared 2D matrix",
			aShape:     tensor.Shape{1, 2, 3},
			a:          []float32{1, 2, 3, 4, 5, 6},
			bShape:     tensor.Shape{3, 2},
			b:          []float32{1, 2, 3, 4, 5, 6},
			wantShape:  tensor.Shape{1, 2, 2},
			wantValues: []float32{22, 28, 49, 64},
		},
		{
			name:       "broadcast head",
			aShape:     tensor.Shape{1, 2, 1, 3},
			a:          []float32{1, 2, 3, 4, 5, 6},
			bShape:     tensor.Shape{1, 1, 3, 2},
			b:          []float32{1, 2, 3, 4, 5, 6},
			wantShape:  tensor.Shape{1, 2, 1, 2},
			wantValues: []float32{22, 28, 49, 64},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newFloat32Tensor(t, test.aShape, test.a)
			b := newFloat32Tensor(t, test.bShape, test.b)
			gotTensor := backend.BatchMatMul(a, b)
			if !gotTensor.Shape().Equal(test.wantShape) {
				t.Fatalf("shape = %v, want %v", gotTensor.Shape(), test.wantShape)
			}
			got := gotTensor.AsFloat32()
			for i := range test.wantValues {
				if math.Abs(float64(got[i]-test.wantValues[i])) > 1e-5 {
					t.Fatalf("result = %v, want %v", got, test.wantValues)
				}
			}
		})
	}
}

func TestLayerNormPrimitivesGPU(t *testing.T) {
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}
	backend, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer backend.Release()

	values := make([]float32, 13*64)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.17) + float64(i%7)*0.1)
	}
	input := newFloat32Tensor(t, tensor.Shape{1, 13, 64}, values)
	mean := backend.MeanDim(input, 2, true)
	centered := backend.Sub(input, mean)
	squared := backend.Mul(centered, centered)
	varianceSum := backend.SumDim(squared, 2, true)
	variance := backend.MulScalar(varianceSum, float32(1.0/64.0))
	normalized := backend.Mul(centered, backend.Rsqrt(backend.AddScalar(variance, float32(1e-5))))

	meanValues := mean.AsFloat32()
	centeredValues := centered.AsFloat32()
	squaredValues := squared.AsFloat32()
	varianceSumValues := varianceSum.AsFloat32()
	varianceValues := variance.AsFloat32()
	normalizedValues := normalized.AsFloat32()
	for row := 0; row < 13; row++ {
		var wantMean, wantVariance float64
		for column := 0; column < 64; column++ {
			wantMean += float64(values[row*64+column])
		}
		wantMean /= 64
		for column := 0; column < 64; column++ {
			difference := float64(values[row*64+column]) - wantMean
			wantVariance += difference * difference
		}
		wantVariance /= 64
		if math.Abs(float64(meanValues[row])-wantMean) > 1e-5 {
			t.Fatalf("mean row %d = %v, want %v", row, meanValues[row], wantMean)
		}
		for column := 0; column < 64; column++ {
			wantCentered := float64(values[row*64+column]) - wantMean
			if math.Abs(float64(centeredValues[row*64+column])-wantCentered) > 1e-5 {
				t.Fatalf("centered row %d column %d = %v, want %v", row, column, centeredValues[row*64+column], wantCentered)
			}
			wantSquared := wantCentered * wantCentered
			if math.Abs(float64(squaredValues[row*64+column])-wantSquared) > 1e-5 {
				t.Fatalf("squared row %d column %d = %v, want %v", row, column, squaredValues[row*64+column], wantSquared)
			}
		}
		if math.Abs(float64(varianceValues[row])-wantVariance) > 1e-5 {
			t.Fatalf("variance row %d = %v, want %v; all sums=%v variances=%v squared[240:280]=%v", row, varianceValues[row], wantVariance, varianceSumValues, varianceValues, squaredValues[240:280])
		}
		var normalizedMean, normalizedSquares float64
		for column := 0; column < 64; column++ {
			value := float64(normalizedValues[row*64+column])
			normalizedMean += value
			normalizedSquares += value * value
		}
		normalizedMean /= 64
		normalizedRMS := math.Sqrt(normalizedSquares / 64)
		if math.Abs(normalizedMean) > 1e-4 || math.Abs(normalizedRMS-1) > 1e-3 {
			t.Fatalf("normalized row %d mean=%v RMS=%v", row, normalizedMean, normalizedRMS)
		}
	}
}

func TestMatMulWideGPU(t *testing.T) {
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}
	backend, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	defer backend.Release()

	const rows, reduction, columns = 1, 192, 128
	aValues := make([]float32, rows*reduction)
	bValues := make([]float32, reduction*columns)
	for i := range aValues {
		aValues[i] = float32(math.Sin(float64(i)*0.13) * 0.7)
	}
	for i := range bValues {
		bValues[i] = float32(math.Cos(float64(i)*0.07) * 0.1)
	}
	a := newFloat32Tensor(t, tensor.Shape{rows, reduction}, aValues)
	b := newFloat32Tensor(t, tensor.Shape{reduction, columns}, bValues)
	got := backend.MatMul(a, b).AsFloat32()
	for column := 0; column < columns; column++ {
		var want float32
		for k := 0; k < reduction; k++ {
			want += aValues[k] * bValues[k*columns+column]
		}
		if math.Abs(float64(got[column]-want)) > 1e-5 {
			t.Fatalf("column %d = %v, want %v", column, got[column], want)
		}
	}

	bOriginalValues := make([]float32, len(bValues))
	for k := 0; k < reduction; k++ {
		for column := 0; column < columns; column++ {
			bOriginalValues[column*reduction+k] = bValues[k*columns+column]
		}
	}
	bOriginal := newFloat32Tensor(t, tensor.Shape{columns, reduction}, bOriginalValues)
	transposed := backend.Transpose(bOriginal, 1, 0)
	got = backend.MatMul(a, transposed).AsFloat32()
	for column := 0; column < columns; column++ {
		var want float32
		for k := 0; k < reduction; k++ {
			want += aValues[k] * bValues[k*columns+column]
		}
		if math.Abs(float64(got[column]-want)) > 1e-5 {
			t.Fatalf("lazy transpose column %d = %v, want %v", column, got[column], want)
		}
	}

	biasValues := make([]float32, columns)
	for i := range biasValues {
		biasValues[i] = float32(i%11)*0.01 - 0.05
	}
	bias := newFloat32Tensor(t, tensor.Shape{columns}, biasValues)
	withBias := backend.Add(backend.MatMul(a, backend.Transpose(bOriginal, 1, 0)), bias).AsFloat32()
	for column := 0; column < columns; column++ {
		var want float32
		for k := 0; k < reduction; k++ {
			want += aValues[k] * bValues[k*columns+column]
		}
		want += biasValues[column]
		if math.Abs(float64(withBias[column]-want)) > 1e-5 {
			t.Fatalf("lazy Gemm column %d = %v, want %v", column, withBias[column], want)
		}
	}
}

func newFloat32Tensor(t *testing.T, shape tensor.Shape, values []float32) *tensor.RawTensor {
	t.Helper()
	result, err := tensor.NewRaw(shape, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(result.AsFloat32(), values)
	return result
}

func referenceConv1D(input, kernel, bias []float32, inputShape, kernelShape tensor.Shape, stride, padLeft, padRight, dilation, groups int) []float32 {
	batch, inChannels, inputLength := inputShape[0], inputShape[1], inputShape[2]
	outChannels, inChannelsPerGroup, kernelLength := kernelShape[0], kernelShape[1], kernelShape[2]
	outputLength := (inputLength+padLeft+padRight-dilation*(kernelLength-1)-1)/stride + 1
	result := make([]float32, batch*outChannels*outputLength)
	outChannelsPerGroup := outChannels / groups
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for outputChannel := 0; outputChannel < outChannels; outputChannel++ {
			group := outputChannel / outChannelsPerGroup
			for outputPosition := 0; outputPosition < outputLength; outputPosition++ {
				var sum float32
				if bias != nil {
					sum = bias[outputChannel]
				}
				for inputChannelInGroup := 0; inputChannelInGroup < inChannelsPerGroup; inputChannelInGroup++ {
					inputChannel := group*inChannelsPerGroup + inputChannelInGroup
					for kernelPosition := 0; kernelPosition < kernelLength; kernelPosition++ {
						inputPosition := outputPosition*stride - padLeft + kernelPosition*dilation
						if inputPosition >= 0 && inputPosition < inputLength {
							inputIndex := (batchIndex*inChannels+inputChannel)*inputLength + inputPosition
							kernelIndex := (outputChannel*inChannelsPerGroup+inputChannelInGroup)*kernelLength + kernelPosition
							sum += input[inputIndex] * kernel[kernelIndex]
						}
					}
				}
				result[(batchIndex*outChannels+outputChannel)*outputLength+outputPosition] = sum
			}
		}
	}
	return result
}
