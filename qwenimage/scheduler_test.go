package qwenimage

import (
	"math"
	"testing"
)

func TestSigmasTurbo(t *testing.T) {
	t.Parallel()
	got := Sigmas(8)
	if len(got) != 9 {
		t.Fatalf("length %d != 9", len(got))
	}
	if got[0] != 1 || got[8] != 0 {
		t.Fatalf("endpoints: %v", got)
	}
	// Hand-computed second value: base s=0.875, e^1.15=3.1581929096897835,
	// sigma = e^mu / (e^mu + 1/7) = 0.9567236...
	want1 := 3.1581929096897835 / (3.1581929096897835 + 1.0/7.0)
	if math.Abs(got[1]-want1) > 1e-12 {
		t.Fatalf("sigma[1]=%v want %v", got[1], want1)
	}
	for i := 1; i < len(got); i++ {
		if got[i] >= got[i-1] {
			t.Fatalf("not strictly decreasing at %d: %v", i, got)
		}
	}
	// Cross-checked against diffusers FlowMatchEulerDiscreteScheduler
	// (set_timesteps(sigmas=linspace(1, 1/8, 8), mu=1.15)) on 2026-07-18.
	diffusers := []float64{1.0, 0.95672369, 0.9045307636, 0.8403487802,
		0.7595109344, 0.6545667648, 0.5128441453, 0.3109010756, 0.0}
	for i, want := range diffusers {
		if math.Abs(got[i]-want) > 1e-7 {
			t.Fatalf("sigma[%d]=%v, diffusers %v", i, got[i], want)
		}
	}
}

func TestEulerStepAndNoise(t *testing.T) {
	t.Parallel()
	lat := []float32{1, 2}
	vel := []float32{10, -10}
	EulerStep(lat, vel, 1.0, 0.5) // d = -0.5
	if lat[0] != -4 || lat[1] != 7 {
		t.Fatalf("euler step got %v", lat)
	}

	a := NoiseLatents(42, 8)
	b := NoiseLatents(42, 8)
	c := NoiseLatents(43, 8)
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("same seed differs")
		}
	}
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
		}
	}
	if same {
		t.Fatal("different seeds identical")
	}
}
