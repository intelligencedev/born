package qwenimage

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// TextEncoderConfig describes the Qwen3-VL text tower. Defaults are pinned
// from the Qwen3VL-4B-Instruct GGUF metadata (see reference/ORACLE.md).
type TextEncoderConfig struct {
	Layers    int
	Hidden    int
	Heads     int
	KVHeads   int
	HeadDim   int
	FFN       int
	Vocab     int
	Eps       float32
	RopeTheta float64
}

// DefaultTextEncoderConfig matches Qwen3-VL-4B-Instruct.
func DefaultTextEncoderConfig() TextEncoderConfig {
	return TextEncoderConfig{
		Layers: 36, Hidden: 2560, Heads: 32, KVHeads: 8, HeadDim: 128,
		FFN: 9728, Vocab: 151936, Eps: 1e-6, RopeTheta: 5_000_000,
	}
}

// WeightSource yields dequantized f32 tensors by GGUF (llama.cpp) name.
// Implemented by the gguf TensorConverter and by test fixture directories.
type WeightSource interface {
	LoadF32(name string) ([]float32, []int, error)
}

type teLayer struct {
	attnNorm, ffnNorm []float32 // [hidden]
	qNorm, kNorm      []float32 // [headDim]
	// Projection weights stored transposed as [in, out] row-major so that
	// y[t] = x[t] · W is a straight walk over rows.
	wq, wk, wv, wo             []float32
	ffnGate, ffnUp, ffnDown    []float32
	wqOut, wkOut, wvOut, woOut int
	gateOut, upOut, downOut    int
}

// TextEncoder is the pure-Go Qwen3-VL text tower used as the Krea 2 prompt
// embedding extractor (hidden states only; no LM head).
type TextEncoder struct {
	cfg    TextEncoderConfig
	embed  []float32 // [vocab, hidden]
	layers []teLayer
}

// LoadTextEncoder builds the encoder from ws using GGUF tensor names.
// Strict: any missing tensor or unexpected shape is an error.
func LoadTextEncoder(cfg TextEncoderConfig, ws WeightSource) (*TextEncoder, error) {
	te := &TextEncoder{cfg: cfg}

	var err error
	load := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		data, shape, e := ws.LoadF32(name)
		if e != nil {
			err = fmt.Errorf("load %s: %w", name, e)
			return nil
		}
		if len(want) > 0 && !equalShapeInts(shape, want) {
			err = fmt.Errorf("tensor %s: shape %v, want %v", name, shape, want)
			return nil
		}
		return data
	}
	// transposeToInOut converts a torch-layout [out, in] weight into [in, out].
	transposeToInOut := func(w []float32, out, in int) []float32 {
		t := make([]float32, len(w))
		for o := 0; o < out; o++ {
			for i := 0; i < in; i++ {
				t[i*out+o] = w[o*in+i]
			}
		}
		return t
	}

	te.embed = load("token_embd.weight", cfg.Vocab, cfg.Hidden)
	te.layers = make([]teLayer, cfg.Layers)
	qOut := cfg.Heads * cfg.HeadDim
	kvOut := cfg.KVHeads * cfg.HeadDim
	for i := range te.layers {
		p := fmt.Sprintf("blk.%d.", i)
		l := &te.layers[i]
		l.attnNorm = load(p+"attn_norm.weight", cfg.Hidden)
		l.ffnNorm = load(p+"ffn_norm.weight", cfg.Hidden)
		l.qNorm = load(p+"attn_q_norm.weight", cfg.HeadDim)
		l.kNorm = load(p+"attn_k_norm.weight", cfg.HeadDim)
		l.wq = transposeToInOut(load(p+"attn_q.weight", qOut, cfg.Hidden), qOut, cfg.Hidden)
		l.wk = transposeToInOut(load(p+"attn_k.weight", kvOut, cfg.Hidden), kvOut, cfg.Hidden)
		l.wv = transposeToInOut(load(p+"attn_v.weight", kvOut, cfg.Hidden), kvOut, cfg.Hidden)
		l.wo = transposeToInOut(load(p+"attn_output.weight", cfg.Hidden, qOut), cfg.Hidden, qOut)
		l.ffnGate = transposeToInOut(load(p+"ffn_gate.weight", cfg.FFN, cfg.Hidden), cfg.FFN, cfg.Hidden)
		l.ffnUp = transposeToInOut(load(p+"ffn_up.weight", cfg.FFN, cfg.Hidden), cfg.FFN, cfg.Hidden)
		l.ffnDown = transposeToInOut(load(p+"ffn_down.weight", cfg.Hidden, cfg.FFN), cfg.Hidden, cfg.FFN)
		l.wqOut, l.wkOut, l.wvOut, l.woOut = qOut, kvOut, kvOut, cfg.Hidden
		l.gateOut, l.upOut, l.downOut = cfg.FFN, cfg.FFN, cfg.Hidden
		if err != nil {
			return nil, err
		}
	}
	if err != nil {
		return nil, err
	}
	return te, nil
}

// Forward runs the text tower over one sequence and returns the hidden states
// AFTER each requested layer, HF hidden_states indexing: index L = output of
// block L-1 (L=0, the raw embeddings, is not supported here).
//
// valid marks non-padding tokens. Position ids are the cumulative count of
// valid tokens (pads consume no position) per the official Krea2Pipeline;
// attention is causal and padding keys are masked out.
func (te *TextEncoder) Forward(ids []int32, valid []bool, selectLayers []int) ([][]float32, error) {
	cfg := te.cfg
	seq := len(ids)
	if len(valid) != seq {
		return nil, fmt.Errorf("valid mask length %d != ids %d", len(valid), seq)
	}
	sel := make(map[int]int, len(selectLayers)) // layer index -> output slot
	for slot, l := range selectLayers {
		if l < 1 || l > cfg.Layers {
			return nil, fmt.Errorf("select layer %d out of range [1,%d]", l, cfg.Layers)
		}
		sel[l] = slot
	}
	out := make([][]float32, len(selectLayers))

	// Embedding lookup.
	x := make([]float32, seq*cfg.Hidden)
	for t, id := range ids {
		if id < 0 || int(id) >= cfg.Vocab {
			return nil, fmt.Errorf("token id %d out of vocab", id)
		}
		copy(x[t*cfg.Hidden:], te.embed[int(id)*cfg.Hidden:(int(id)+1)*cfg.Hidden])
	}

	// Cumulative-valid positions.
	positions := make([]int, seq)
	pos := 0
	for t := range positions {
		if valid[t] {
			positions[t] = pos
			pos++
		} else {
			if pos == 0 {
				positions[t] = 0
			} else {
				positions[t] = pos - 1
			}
		}
	}

	// Rope tables per token.
	half := cfg.HeadDim / 2
	cosTab := make([]float32, seq*half)
	sinTab := make([]float32, seq*half)
	for t := 0; t < seq; t++ {
		for i := 0; i < half; i++ {
			freq := math.Pow(cfg.RopeTheta, -2*float64(i)/float64(cfg.HeadDim))
			angle := float64(positions[t]) * freq
			cosTab[t*half+i] = float32(math.Cos(angle))
			sinTab[t*half+i] = float32(math.Sin(angle))
		}
	}

	scratch := newTeScratch(cfg, seq)
	for li := range te.layers {
		l := &te.layers[li]
		te.layerForward(l, x, valid, cosTab, sinTab, scratch)
		if slot, ok := sel[li+1]; ok {
			cp := make([]float32, len(x))
			copy(cp, x)
			out[slot] = cp
		}
	}
	return out, nil
}

type teScratch struct {
	normed, q, k, v, attnOut, proj []float32
	gate, up                       []float32
}

func newTeScratch(cfg TextEncoderConfig, seq int) *teScratch {
	return &teScratch{
		normed:  make([]float32, seq*cfg.Hidden),
		q:       make([]float32, seq*cfg.Heads*cfg.HeadDim),
		k:       make([]float32, seq*cfg.KVHeads*cfg.HeadDim),
		v:       make([]float32, seq*cfg.KVHeads*cfg.HeadDim),
		attnOut: make([]float32, seq*cfg.Heads*cfg.HeadDim),
		proj:    make([]float32, seq*cfg.Hidden),
		gate:    make([]float32, seq*cfg.FFN),
		up:      make([]float32, seq*cfg.FFN),
	}
}

func (te *TextEncoder) layerForward(l *teLayer, x []float32, valid []bool, cosTab, sinTab []float32, s *teScratch) {
	cfg := te.cfg
	seq := len(x) / cfg.Hidden
	half := cfg.HeadDim / 2

	// --- Attention sublayer ---
	rmsNormInto(s.normed, x, l.attnNorm, cfg.Hidden, cfg.Eps)
	matmulInto(s.q, s.normed, l.wq, seq, cfg.Hidden, l.wqOut)
	matmulInto(s.k, s.normed, l.wk, seq, cfg.Hidden, l.wkOut)
	matmulInto(s.v, s.normed, l.wv, seq, cfg.Hidden, l.wvOut)

	// Per-head QK RMSNorm then rotate-half RoPE.
	applyHeadNormRope(s.q, seq, cfg.Heads, cfg.HeadDim, l.qNorm, cfg.Eps, cosTab, sinTab, half)
	applyHeadNormRope(s.k, seq, cfg.KVHeads, cfg.HeadDim, l.kNorm, cfg.Eps, cosTab, sinTab, half)

	// Causal GQA attention with padding-key masking, parallel over heads.
	group := cfg.Heads / cfg.KVHeads
	scale := 1 / float32(math.Sqrt(float64(cfg.HeadDim)))
	var wg sync.WaitGroup
	for h := 0; h < cfg.Heads; h++ {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()
			scores := make([]float32, seq)
			kvh := h / group
			for tq := 0; tq < seq; tq++ {
				qv := s.q[(tq*cfg.Heads+h)*cfg.HeadDim:][:cfg.HeadDim]
				maxScore := float32(math.Inf(-1))
				for tk := 0; tk <= tq; tk++ {
					if !valid[tk] {
						scores[tk] = float32(math.Inf(-1))
						continue
					}
					kv := s.k[(tk*cfg.KVHeads+kvh)*cfg.HeadDim:][:cfg.HeadDim]
					var dot float32
					for d := 0; d < cfg.HeadDim; d++ {
						dot += qv[d] * kv[d]
					}
					dot *= scale
					scores[tk] = dot
					if dot > maxScore {
						maxScore = dot
					}
				}
				outRow := s.attnOut[(tq*cfg.Heads+h)*cfg.HeadDim:][:cfg.HeadDim]
				for d := range outRow {
					outRow[d] = 0
				}
				if math.IsInf(float64(maxScore), -1) {
					continue // no attendable key (all-pad prefix)
				}
				var denom float32
				for tk := 0; tk <= tq; tk++ {
					sc := scores[tk]
					if math.IsInf(float64(sc), -1) {
						scores[tk] = 0
						continue
					}
					e := float32(math.Exp(float64(sc - maxScore)))
					scores[tk] = e
					denom += e
				}
				inv := 1 / denom
				for tk := 0; tk <= tq; tk++ {
					w := scores[tk] * inv
					if w == 0 {
						continue
					}
					vv := s.v[(tk*cfg.KVHeads+kvh)*cfg.HeadDim:][:cfg.HeadDim]
					for d := 0; d < cfg.HeadDim; d++ {
						outRow[d] += w * vv[d]
					}
				}
			}
		}(h)
	}
	wg.Wait()
	matmulInto(s.proj, s.attnOut, l.wo, seq, cfg.Heads*cfg.HeadDim, l.woOut)
	for i := range x {
		x[i] += s.proj[i]
	}

	// --- MLP sublayer (SwiGLU) ---
	rmsNormInto(s.normed, x, l.ffnNorm, cfg.Hidden, cfg.Eps)
	matmulInto(s.gate, s.normed, l.ffnGate, seq, cfg.Hidden, l.gateOut)
	matmulInto(s.up, s.normed, l.ffnUp, seq, cfg.Hidden, l.upOut)
	for i := range s.gate {
		g := float64(s.gate[i])
		s.gate[i] = float32(g/(1+math.Exp(-g))) * s.up[i]
	}
	matmulInto(s.proj, s.gate, l.ffnDown, seq, cfg.FFN, l.downOut)
	for i := range x {
		x[i] += s.proj[i]
	}
}

// rmsNormInto computes standard RMSNorm (weight multiply, HF Qwen3 style).
func rmsNormInto(dst, x, weight []float32, dim int, eps float32) {
	rows := len(x) / dim
	for r := 0; r < rows; r++ {
		row := x[r*dim : (r+1)*dim]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := float32(1 / math.Sqrt(ss/float64(dim)+float64(eps)))
		out := dst[r*dim : (r+1)*dim]
		for i, v := range row {
			out[i] = v * inv * weight[i]
		}
	}
}

// applyHeadNormRope applies per-head RMSNorm then rotate-half RoPE in place.
func applyHeadNormRope(qk []float32, seq, heads, headDim int, normW []float32, eps float32, cosTab, sinTab []float32, half int) {
	for t := 0; t < seq; t++ {
		for h := 0; h < heads; h++ {
			vec := qk[(t*heads+h)*headDim:][:headDim]
			var ss float64
			for _, v := range vec {
				ss += float64(v) * float64(v)
			}
			inv := float32(1 / math.Sqrt(ss/float64(headDim)+float64(eps)))
			for i := range vec {
				vec[i] *= inv * normW[i]
			}
			// rotate-half: pair (i, i+half).
			for i := 0; i < half; i++ {
				c, s := cosTab[t*half+i], sinTab[t*half+i]
				a, b := vec[i], vec[i+half]
				vec[i] = a*c - b*s
				vec[i+half] = b*c + a*s
			}
		}
	}
}

// matmulInto computes dst[seq, out] = x[seq, in] · w[in, out], parallelized
// over sequence rows (the same row-split strategy that paid off in the
// Supertonic conv/batchmatmul work).
func matmulInto(dst, x, w []float32, seq, in, out int) {
	workers := runtime.GOMAXPROCS(0)
	if workers > seq {
		workers = seq
	}
	if workers <= 1 {
		matmulRows(dst, x, w, 0, seq, in, out)
		return
	}
	var wg sync.WaitGroup
	chunk := (seq + workers - 1) / workers
	for start := 0; start < seq; start += chunk {
		end := start + chunk
		if end > seq {
			end = seq
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			matmulRows(dst, x, w, start, end, in, out)
		}(start, end)
	}
	wg.Wait()
}

func matmulRows(dst, x, w []float32, from, to, in, out int) {
	for t := from; t < to; t++ {
		row := x[t*in : (t+1)*in]
		o := dst[t*out : (t+1)*out]
		for i := range o {
			o[i] = 0
		}
		for i, xv := range row {
			if xv == 0 {
				continue
			}
			wRow := w[i*out : (i+1)*out]
			for j, wv := range wRow {
				o[j] += xv * wv
			}
		}
	}
}

func equalShapeInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
