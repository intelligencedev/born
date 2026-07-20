//go:build darwin || linux || windows

package moonshine

import (
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/intelligencedev/born/backend/cpu"
)

func TestTranscribeWebGPUMatchesCPU(t *testing.T) {
	modelDir := os.Getenv("MOONSHINE_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set MOONSHINE_MODEL_DIR")
	}
	gpuSTT, err := New(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuSTT.Close()
	if gpuSTT.BackendName() == "CPU" {
		t.Skip("hardware WebGPU unavailable; correctly fell back to CPU")
	}
	if !strings.HasPrefix(gpuSTT.BackendName(), "WebGPU") {
		t.Fatalf("unexpected backend %q", gpuSTT.BackendName())
	}
	cpuSTT, err := newWithBackend(modelDir, cpu.New())
	if err != nil {
		t.Fatal(err)
	}

	audio := syntheticAudio()
	if wavPath := os.Getenv("MOONSHINE_WAV"); wavPath != "" {
		audio, err = readPCM16Mono16k(wavPath)
		if err != nil {
			t.Fatal(err)
		}
	}
	started := time.Now()
	want, err := cpuSTT.TranscribeTokens(audio)
	if err != nil {
		t.Fatal(err)
	}
	cpuDuration := time.Since(started)
	started = time.Now()
	got, err := gpuSTT.TranscribeTokens(audio)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("GPU ids %v, CPU ids %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: GPU=%d CPU=%d; GPU ids %v CPU ids %v", i, got[i], want[i], got, want)
		}
	}
	t.Logf("%s matched CPU tokens %v; CPU=%s GPU=%s", gpuSTT.BackendName(), got, cpuDuration, time.Since(started))
}

func syntheticAudio() []float32 {
	audio := make([]float32, SampleRate)
	for i := range audio {
		seconds := float64(i) / SampleRate
		audio[i] = float32(0.15*math.Sin(2*math.Pi*220*seconds) + 0.05*math.Sin(2*math.Pi*440*seconds))
	}
	return audio
}

func readPCM16Mono16k(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 44 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, os.ErrInvalid
	}
	var pcm []byte
	for offset := 12; offset+8 <= len(data); {
		size := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		start := offset + 8
		end := start + size
		if end > len(data) {
			return nil, os.ErrInvalid
		}
		switch string(data[offset : offset+4]) {
		case "fmt ":
			if size < 16 || binary.LittleEndian.Uint16(data[start:start+2]) != 1 ||
				binary.LittleEndian.Uint16(data[start+2:start+4]) != 1 ||
				binary.LittleEndian.Uint32(data[start+4:start+8]) != SampleRate ||
				binary.LittleEndian.Uint16(data[start+14:start+16]) != 16 {
				return nil, os.ErrInvalid
			}
		case "data":
			pcm = data[start:end]
		}
		offset = end + size%2
	}
	if len(pcm) == 0 {
		return nil, os.ErrInvalid
	}
	audio := make([]float32, len(pcm)/2)
	for i := range audio {
		audio[i] = float32(int16(binary.LittleEndian.Uint16(pcm[i*2:]))) / 32768
	}
	return audio, nil
}
