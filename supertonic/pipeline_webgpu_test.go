//go:build darwin || linux || windows

package supertonic

import (
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/intelligencedev/born/backend/cpu"
	"github.com/intelligencedev/born/internal/tensor"
)

func TestSynthesizeWebGPU(t *testing.T) {
	modelDir := os.Getenv("SUPERTONIC_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set SUPERTONIC_MODEL_DIR")
	}

	tts, err := New(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer tts.Close()
	if !strings.HasPrefix(tts.BackendName(), "WebGPU") {
		t.Skipf("hardware WebGPU unavailable; selected %s", tts.BackendName())
	}

	started := time.Now()
	samples, err := tts.Synthesize("Hi.", "M1", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 {
		t.Fatal("synthesis returned no samples")
	}
	for i, sample := range samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			t.Fatalf("sample %d is not finite: %v", i, sample)
		}
	}
	firstDuration := time.Since(started)
	started = time.Now()
	secondSamples, err := tts.Synthesize("Hi.", "M1", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(secondSamples) != len(samples) {
		t.Fatalf("second synthesis returned %d samples, first returned %d", len(secondSamples), len(samples))
	}
	t.Logf("%s synthesized %d samples in %s cold and %s warm", tts.BackendName(), len(samples), firstDuration, time.Since(started))
}

func TestDurationPredictorWebGPUMatchesCPU(t *testing.T) {
	modelDir := os.Getenv("SUPERTONIC_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set SUPERTONIC_MODEL_DIR")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gpuTTS, err := New(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuTTS.Close()
	if !strings.HasPrefix(gpuTTS.BackendName(), "WebGPU") {
		t.Skipf("hardware WebGPU unavailable; selected %s", gpuTTS.BackendName())
	}
	cpuTTS, err := newWithBackend(modelDir, cpu.New())
	if err != nil {
		t.Fatal(err)
	}

	run := func(tts *TTS) *tensor.RawTensor {
		pre, ok := preprocessText("Hi.", "en")
		if !ok {
			t.Fatal("preprocess failed")
		}
		ids, mask := tokenize(tts.indexer, pre)
		style, styleErr := tts.style("M1")
		if styleErr != nil {
			t.Fatal(styleErr)
		}
		textIDs, _ := rawI64([]int{1, len(ids)}, ids)
		textMask, _ := rawF32([]int{1, 1, len(ids)}, mask)
		styleDP, _ := rawF32(style.DPDims, style.DPData)
		output, forwardErr := tts.dp.ForwardNamed(map[string]*tensor.RawTensor{
			"text_ids": textIDs, "style_dp": styleDP, "text_mask": textMask,
		})
		if forwardErr != nil {
			t.Fatal(forwardErr)
		}
		return output["duration"]
	}

	want := run(cpuTTS).AsFloat32()[0]
	got := run(gpuTTS).AsFloat32()[0]
	if math.IsNaN(float64(got)) || math.Abs(float64(got-want)) > 1e-2 {
		t.Fatalf("duration = %v, CPU = %v", got, want)
	}
}

func TestSynthesizeWebGPUMatchesCPU(t *testing.T) {
	modelDir := os.Getenv("SUPERTONIC_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set SUPERTONIC_MODEL_DIR")
	}
	gpuTTS, err := New(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuTTS.Close()
	if !strings.HasPrefix(gpuTTS.BackendName(), "WebGPU") {
		t.Skipf("hardware WebGPU unavailable; selected %s", gpuTTS.BackendName())
	}
	cpuTTS, err := newWithBackend(modelDir, cpu.New())
	if err != nil {
		t.Fatal(err)
	}

	random := rand.New(rand.NewSource(42))
	noise := make([]float32, cpuTTS.cfg.LatentDim*cpuTTS.cfg.ChunkCompressFactor*4096)
	for i := range noise {
		noise[i] = random.Float32()*2 - 1
	}
	options := Options{injectNoise: noise}
	want, err := cpuTTS.Synthesize("Hi.", "M1", options)
	if err != nil {
		t.Fatal(err)
	}
	got, err := gpuTTS.Synthesize("Hi.", "M1", options)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("GPU returned %d samples, CPU returned %d", len(got), len(want))
	}
	var maxAbsoluteDifference float64
	for i := range want {
		difference := math.Abs(float64(got[i] - want[i]))
		if difference > maxAbsoluteDifference {
			maxAbsoluteDifference = difference
		}
	}
	if maxAbsoluteDifference > 3e-3 {
		t.Fatalf("GPU waveform differs from CPU: max absolute difference %g", maxAbsoluteDifference)
	}
	t.Logf("GPU waveform matches CPU over %d samples (max abs %g)", len(got), maxAbsoluteDifference)
}

func TestSupertonicModelsWebGPUMatchCPU(t *testing.T) {
	modelDir := os.Getenv("SUPERTONIC_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set SUPERTONIC_MODEL_DIR")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gpuTTS, err := New(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuTTS.Close()
	if !strings.HasPrefix(gpuTTS.BackendName(), "WebGPU") {
		t.Skipf("hardware WebGPU unavailable; selected %s", gpuTTS.BackendName())
	}
	cpuTTS, err := newWithBackend(modelDir, cpu.New())
	if err != nil {
		t.Fatal(err)
	}

	assertClose := func(name string, got, want *tensor.RawTensor, tolerance float64) {
		t.Helper()
		if !got.Shape().Equal(want.Shape()) {
			t.Fatalf("%s shape: GPU=%v CPU=%v", name, got.Shape(), want.Shape())
		}
		gotValues, wantValues := got.AsFloat32(), want.AsFloat32()
		var maxDifference float64
		for i := range wantValues {
			difference := math.Abs(float64(gotValues[i] - wantValues[i]))
			if math.IsNaN(float64(gotValues[i])) || difference > tolerance {
				t.Fatalf("%s first differs at %d: GPU=%v CPU=%v (difference %g)", name, i, gotValues[i], wantValues[i], difference)
			}
			if difference > maxDifference {
				maxDifference = difference
			}
		}
		t.Logf("%s max abs difference %g", name, maxDifference)
	}

	pre, _ := preprocessText("Hi.", "en")
	ids, mask := tokenize(cpuTTS.indexer, pre)
	style, err := cpuTTS.style("M1")
	if err != nil {
		t.Fatal(err)
	}
	textIDs, _ := rawI64([]int{1, len(ids)}, ids)
	textMask, _ := rawF32([]int{1, 1, len(ids)}, mask)
	styleTTL, _ := rawF32(style.TTLDims, style.TTLData)
	textInputs := map[string]*tensor.RawTensor{
		"text_ids": textIDs, "style_ttl": styleTTL, "text_mask": textMask,
	}
	cpuText, err := cpuTTS.textEnc.ForwardNamed(textInputs)
	if err != nil {
		t.Fatal(err)
	}
	gpuText, err := gpuTTS.textEnc.ForwardNamed(textInputs)
	if err != nil {
		t.Fatal(err)
	}
	assertClose("text encoder", gpuText["text_emb"], cpuText["text_emb"], 2e-3)

	latentLength := 12
	latentDimensions := cpuTTS.cfg.LatentDim * cpuTTS.cfg.ChunkCompressFactor
	random := rand.New(rand.NewSource(7))
	latentValues := make([]float32, latentDimensions*latentLength)
	for i := range latentValues {
		latentValues[i] = random.Float32()*2 - 1
	}
	latent, _ := rawF32([]int{1, latentDimensions, latentLength}, latentValues)
	latentMaskValues := make([]float32, latentLength)
	for i := range latentMaskValues {
		latentMaskValues[i] = 1
	}
	latentMask, _ := rawF32([]int{1, 1, latentLength}, latentMaskValues)
	currentStep, _ := rawF32([]int{1}, []float32{0})
	totalStep, _ := rawF32([]int{1}, []float32{1})
	vectorInputs := map[string]*tensor.RawTensor{
		"noisy_latent": latent, "text_emb": cpuText["text_emb"], "style_ttl": styleTTL,
		"latent_mask": latentMask, "text_mask": textMask,
		"current_step": currentStep, "total_step": totalStep,
	}
	cpuVector, err := cpuTTS.vectorEst.ForwardNamed(vectorInputs)
	if err != nil {
		t.Fatal(err)
	}
	gpuVector, err := gpuTTS.vectorEst.ForwardNamed(vectorInputs)
	if err != nil {
		t.Fatal(err)
	}
	assertClose("vector estimator", gpuVector["denoised_latent"], cpuVector["denoised_latent"], 3e-3)

	vocoderInputs := map[string]*tensor.RawTensor{"latent": latent}
	cpuVocoder, err := cpuTTS.vocoder.ForwardNamed(vocoderInputs)
	if err != nil {
		t.Fatal(err)
	}
	gpuVocoder, err := gpuTTS.vocoder.ForwardNamed(vocoderInputs)
	if err != nil {
		t.Fatal(err)
	}
	assertClose("vocoder", gpuVocoder["wav_tts"], cpuVocoder["wav_tts"], 3e-3)
}
