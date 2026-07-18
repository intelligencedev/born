package qwenimage

import (
	"math"
	mrand "math/rand/v2"
)

// TurboShiftMu is the constant flow-shift for the distilled Krea 2 Turbo
// (pipeline_krea2.py: `if self.config.is_distilled: mu = 1.15`).
const TurboShiftMu = 1.15

// Sigmas returns the flow-matching sigma schedule for the distilled Turbo
// model: base sigmas linspace(1, 1/steps, steps), each time-shifted by
// sigma' = e^mu / (e^mu + (1/s - 1)), with a trailing 0. Length steps+1.
func Sigmas(steps int) []float64 {
	emu := math.Exp(TurboShiftMu)
	out := make([]float64, steps+1)
	for i := 0; i < steps; i++ {
		// linspace(1, 1/steps, steps)
		s := 1.0
		if steps > 1 {
			s = 1.0 - float64(i)/float64(steps-1)*(1.0-1.0/float64(steps))
		}
		out[i] = emu / (emu + (1/s - 1))
	}
	out[steps] = 0
	return out
}

// EulerStep advances packed latents in place:
// x <- x + (sigmaNext - sigmaCur) * velocity.
func EulerStep(latents, velocity []float32, sigmaCur, sigmaNext float64) {
	d := float32(sigmaNext - sigmaCur)
	for i := range latents {
		latents[i] += d * velocity[i]
	}
}

// NoiseLatents draws seeded standard-normal packed latents for production
// runs. Parity tests inject harness-dumped noise instead — schedule
// exactness, not RNG equality, is what parity requires.
func NoiseLatents(seed uint64, n int) []float32 {
	rng := mrand.New(mrand.NewPCG(seed, 0))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}
