// Package moonshine runs Moonshine speech-to-text (onnx-community exports) on
// the Born runtime: raw 16 kHz mono float32 audio in, text out. Uses the
// non-merged decoder pair (decoder_model + decoder_with_past_model) with greedy
// decoding and KV-cache reuse.
package moonshine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/intelligencedev/born/backend/cpu"
	"github.com/intelligencedev/born/internal/tensor"
	"github.com/intelligencedev/born/onnx"
)

const (
	SampleRate   = 16000
	startTokenID = 1
	eosTokenID   = 2
	// tokensPerSecond bounds greedy length: Moonshine guidance is ~6.5 tokens/s.
	tokensPerSecond = 6.5
)

type STT struct {
	encoder     onnx.Model
	decoder     onnx.Model
	decoderPast onnx.Model
	pastInputs  []string // decoder_with_past input names (excluding input_ids)
	vocab       []string // id -> token
}

// New loads the three ONNX graphs and tokenizer.json from modelDir
// (onnx/encoder_model.onnx, onnx/decoder_model.onnx,
// onnx/decoder_with_past_model.onnx, tokenizer.json).
func New(modelDir string) (*STT, error) {
	backend := cpu.New()
	opts := onnx.DefaultLoadOptions()
	opts.StrictMode = true
	load := func(name string) (onnx.Model, error) {
		return onnx.Load(filepath.Join(modelDir, "onnx", name), backend, opts)
	}
	enc, err := load("encoder_model.onnx")
	if err != nil {
		return nil, fmt.Errorf("load encoder: %w", err)
	}
	dec, err := load("decoder_model.onnx")
	if err != nil {
		return nil, fmt.Errorf("load decoder: %w", err)
	}
	decP, err := load("decoder_with_past_model.onnx")
	if err != nil {
		return nil, fmt.Errorf("load decoder_with_past: %w", err)
	}
	vocab, err := loadVocab(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	var pastIn []string
	for _, n := range decP.InputNames() {
		if n != "input_ids" {
			pastIn = append(pastIn, n)
		}
	}
	return &STT{encoder: enc, decoder: dec, decoderPast: decP, pastInputs: pastIn, vocab: vocab}, nil
}

func loadVocab(path string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer: %w", err)
	}
	var doc struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	maxID := 0
	for _, id := range doc.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	vocab := make([]string, maxID+1)
	for tok, id := range doc.Model.Vocab {
		vocab[id] = tok
	}
	return vocab, nil
}

// DecodeTokens converts token ids to text: Llama-style SentencePiece decoding
// (skip specials, "▁" -> space, byte-fallback tokens "<0xNN>").
func (s *STT) DecodeTokens(ids []int64) string {
	var b strings.Builder
	for _, id := range ids {
		if id == startTokenID || id == eosTokenID || id < 0 || int(id) >= len(s.vocab) {
			continue
		}
		tok := s.vocab[id]
		if len(tok) == 6 && strings.HasPrefix(tok, "<0x") && strings.HasSuffix(tok, ">") {
			if v, err := strconv.ParseUint(tok[3:5], 16, 8); err == nil {
				b.WriteByte(byte(v))
				continue
			}
		}
		b.WriteString(strings.ReplaceAll(tok, "▁", " "))
	}
	return strings.TrimSpace(b.String())
}

func rawF32(shape []int, data []float32) (*tensor.RawTensor, error) {
	r, err := tensor.NewRaw(tensor.Shape(shape), tensor.Float32, tensor.CPU)
	if err != nil {
		return nil, err
	}
	copy(r.AsFloat32(), data)
	return r, nil
}

func rawI64Scalar(v int64) (*tensor.RawTensor, error) {
	r, err := tensor.NewRaw(tensor.Shape{1, 1}, tensor.Int64, tensor.CPU)
	if err != nil {
		return nil, err
	}
	r.AsInt64()[0] = v
	return r, nil
}

func argmaxLastRow(logits *tensor.RawTensor) int64 {
	d := logits.AsFloat32()
	sh := logits.Shape()
	vocab := sh[len(sh)-1]
	row := d[len(d)-vocab:]
	best, bi := row[0], 0
	for i, v := range row[1:] {
		if v > best {
			best, bi = v, i+1
		}
	}
	return int64(bi)
}

// TranscribeTokens runs greedy decoding and returns the raw token ids
// (including start and eos).
func (s *STT) TranscribeTokens(audio []float32) ([]int64, error) {
	if len(audio) == 0 {
		return nil, fmt.Errorf("moonshine: empty audio")
	}
	in, err := rawF32([]int{1, len(audio)}, audio)
	if err != nil {
		return nil, err
	}
	encOut, err := s.encoder.ForwardNamed(map[string]*tensor.RawTensor{"input_values": in})
	if err != nil {
		return nil, fmt.Errorf("encoder: %w", err)
	}
	hidden := encOut["last_hidden_state"]
	if hidden == nil {
		for _, v := range encOut {
			hidden = v // single-output graph
		}
	}

	maxTokens := int(float64(len(audio))/SampleRate*tokensPerSecond) + 8

	startT, err := rawI64Scalar(startTokenID)
	if err != nil {
		return nil, err
	}
	kv, err := s.decoder.ForwardNamed(map[string]*tensor.RawTensor{
		"input_ids": startT, "encoder_hidden_states": hidden,
	})
	if err != nil {
		return nil, fmt.Errorf("decoder step 0: %w", err)
	}
	ids := []int64{startTokenID, argmaxLastRow(kv["logits"])}

	for len(ids)-1 < maxTokens && ids[len(ids)-1] != eosTokenID {
		lastT, err := rawI64Scalar(ids[len(ids)-1])
		if err != nil {
			return nil, err
		}
		feed := map[string]*tensor.RawTensor{"input_ids": lastT}
		for _, n := range s.pastInputs {
			src := strings.Replace(n, "past_key_values", "present", 1)
			t, ok := kv[src]
			if !ok {
				return nil, fmt.Errorf("moonshine: missing cache tensor %s", src)
			}
			feed[n] = t
		}
		out, err := s.decoderPast.ForwardNamed(feed)
		if err != nil {
			return nil, fmt.Errorf("decoder step %d: %w", len(ids), err)
		}
		// decoder presents are re-emitted each step; encoder presents persist
		// from step 0.
		for n, v := range out {
			kv[n] = v
		}
		ids = append(ids, argmaxLastRow(out["logits"]))
	}
	return ids, nil
}

// Transcribe runs greedy decoding and returns the recognized text.
func (s *STT) Transcribe(audio []float32) (string, error) {
	ids, err := s.TranscribeTokens(audio)
	if err != nil {
		return "", err
	}
	return s.DecodeTokens(ids), nil
}
