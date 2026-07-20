//go:build windows || linux || darwin

package webgpu

import (
	"math"
	"testing"

	"github.com/intelligencedev/born/internal/tensor"
)

func TestPowGPU(t *testing.T) {
	backend := newSTTTestBackend(t)
	base := newFloat32Tensor(t, tensor.Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	scalar := newFloat32Tensor(t, tensor.Shape{1}, []float32{2})
	got, err := backend.Pow(base, scalar)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Close(t, got.AsFloat32(), []float32{1, 4, 9, 16, 25, 36}, 1e-5)

	exponents := newFloat32Tensor(t, tensor.Shape{2, 3}, []float32{1, 2, 3, 0.5, 1, 2})
	got, err = backend.Pow(base, exponents)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32Close(t, got.AsFloat32(), []float32{1, 4, 27, 2, 5, 36}, 1e-5)
}

func TestSliceGPU(t *testing.T) {
	backend := newSTTTestBackend(t)
	input := newFloat32Tensor(t, tensor.Shape{2, 3, 4}, []float32{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	})
	tests := []struct {
		name               string
		starts, ends, axes []int64
		steps              []int64
		wantShape          tensor.Shape
		want               []float32
	}{
		{
			name: "strided dimensions", starts: []int64{0, 1}, ends: []int64{2, 4}, axes: []int64{1, 2}, steps: []int64{1, 2},
			wantShape: tensor.Shape{2, 2, 2}, want: []float32{1, 3, 5, 7, 13, 15, 17, 19},
		},
		{
			name: "reverse last dimension", starts: []int64{-1}, ends: []int64{-5}, axes: []int64{2}, steps: []int64{-1},
			wantShape: tensor.Shape{2, 3, 4}, want: []float32{
				3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8,
				15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := backend.Slice(input, test.starts, test.ends, test.axes, test.steps)
			if err != nil {
				t.Fatal(err)
			}
			if !got.Shape().Equal(test.wantShape) {
				t.Fatalf("shape = %v, want %v", got.Shape(), test.wantShape)
			}
			assertFloat32Close(t, got.AsFloat32(), test.want, 0)
		})
	}
}

func TestInstanceNormalizationGPU(t *testing.T) {
	backend := newSTTTestBackend(t)
	input := newFloat32Tensor(t, tensor.Shape{1, 2, 4}, []float32{1, 2, 3, 4, -2, 0, 2, 4})
	scale := newFloat32Tensor(t, tensor.Shape{2}, []float32{1.5, 0.5})
	bias := newFloat32Tensor(t, tensor.Shape{2}, []float32{-0.25, 1})
	got, err := backend.InstanceNormalization(input, scale, bias, 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	want := make([]float32, 8)
	values := input.AsFloat32()
	for channel := 0; channel < 2; channel++ {
		segment := values[channel*4 : channel*4+4]
		var mean, variance float64
		for _, value := range segment {
			mean += float64(value)
		}
		mean /= 4
		for _, value := range segment {
			difference := float64(value) - mean
			variance += difference * difference
		}
		variance /= 4
		for index, value := range segment {
			want[channel*4+index] = float32((float64(value)-mean)/math.Sqrt(variance+1e-5))*scale.AsFloat32()[channel] + bias.AsFloat32()[channel]
		}
	}
	assertFloat32Close(t, got.AsFloat32(), want, 2e-5)
}

func newSTTTestBackend(t *testing.T) *Backend {
	t.Helper()
	if !computeAvailable {
		t.Skip("WebGPU compute not available")
	}
	backend, err := New()
	if err != nil {
		t.Skipf("WebGPU not available: %v", err)
	}
	t.Cleanup(backend.Release)
	return backend
}

func assertFloat32Close(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if math.Abs(float64(got[index]-want[index])) > tolerance {
			t.Fatalf("value %d = %v, want %v; got %v", index, got[index], want[index], got)
		}
	}
}
