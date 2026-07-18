package qwenimage

import (
	"fmt"
	"math"
	"sync"
)

// DiTConfig describes the Krea 2 single-stream diffusion transformer.
// Defaults pinned from sd.cpp krea2.hpp + diffusers Krea2Transformer2DModel
// (see reference/ORACLE.md).
type DiTConfig struct {
	InChannels  int // packed (patchified) channels: vae_ch * patch^2
	Layers      int
	HeadDim     int
	Heads       int
	KVHeads     int
	FFN         int
	TimestepDim int
	TextDim     int
	TextLayers  int // tapped text-encoder hidden states per token
	TextHeads   int
	TextKVHeads int
	TextFFN     int
	AxesDim     [3]int // rope split across (t, h, w); sums to HeadDim
	RopeTheta   float64
	Eps         float32
}

// DefaultDiTConfig matches Krea 2 Turbo.
func DefaultDiTConfig() DiTConfig {
	return DiTConfig{
		InChannels: 64, Layers: 28, HeadDim: 128, Heads: 48, KVHeads: 12,
		FFN: 16384, TimestepDim: 256, TextDim: 2560, TextLayers: 12,
		TextHeads: 20, TextKVHeads: 20, TextFFN: 6912,
		AxesDim: [3]int{32, 48, 48}, RopeTheta: 1000, Eps: 1e-5,
	}
}

func (c DiTConfig) hidden() int { return c.Heads * c.HeadDim }

// ditAttn holds one Krea attention module (DiT block or text fusion block).
// All linears bias-free, weights stored transposed [in, out].
type ditAttn struct {
	wq, wk, wv, gate, wo halfMat
	qNorm, kNorm         []float32 // zero-centered scales [headDim]
	dim, heads, kvHeads  int
	headDim              int
}

type ditFF struct {
	gate, up, down halfMat // transposed [in, out], f16 bits
	dim, ffn       int
}

type fusionBlock struct {
	preNorm, postNorm []float32 // zero-centered [dim]
	attn              ditAttn
	ff                ditFF
}

type ditBlock struct {
	modLin            []float32 // [6*hidden] learned additive table
	preNorm, postNorm []float32 // zero-centered [hidden]
	attn              ditAttn
	ff                ditFF
}

// DiT is the pure-Go Krea 2 diffusion transformer.
type DiT struct {
	cfg DiTConfig

	imgInW   halfMat
	imgInB   []float32 // [hidden]
	tmlp0W   halfMat
	tmlp0B   []float32
	tmlp2W   halfMat
	tmlp2B   []float32
	tprojW   halfMat
	tprojB   []float32
	txtNorm  []float32 // zero-centered [textDim]
	txtMlp1W halfMat
	txtMlp1B []float32
	txtMlp3W halfMat
	txtMlp3B []float32

	fusionLayerwise []fusionBlock
	fusionRefiner   []fusionBlock
	projector       []float32 // [textLayers] (collapses the layer axis)

	blocks []ditBlock

	lastNorm   []float32 // zero-centered [hidden]
	lastModLin []float32 // [2*hidden]
	lastLinW   halfMat
	lastLB     []float32 // [inCh]
}

// LoadDiT builds the transformer from ws using GGUF tensor names. Strict.
func LoadDiT(cfg DiTConfig, ws WeightSource) (*DiT, error) {
	d := &DiT{cfg: cfg}
	h := cfg.hidden()

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
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		wantN := 1
		for _, dim := range want {
			wantN *= dim
		}
		if len(want) > 0 && n != wantN {
			err = fmt.Errorf("tensor %s: shape %v, want %v", name, shape, want)
			return nil
		}
		return data
	}
	tr := func(w []float32, out, in int) halfMat {
		if w == nil {
			return nil
		}
		t := make([]float32, len(w))
		for o := 0; o < out; o++ {
			for i := 0; i < in; i++ {
				t[i*out+o] = w[o*in+i]
			}
		}
		return toHalf(t)
	}
	loadAttn := func(prefix string, dim, heads, kvHeads, headDim int) ditAttn {
		return ditAttn{
			wq:    tr(load(prefix+"attn.wq.weight", heads*headDim, dim), heads*headDim, dim),
			wk:    tr(load(prefix+"attn.wk.weight", kvHeads*headDim, dim), kvHeads*headDim, dim),
			wv:    tr(load(prefix+"attn.wv.weight", kvHeads*headDim, dim), kvHeads*headDim, dim),
			gate:  tr(load(prefix+"attn.gate.weight", dim, dim), dim, dim),
			wo:    tr(load(prefix+"attn.wo.weight", dim, dim), dim, dim),
			qNorm: load(prefix+"attn.qknorm.qnorm.scale", headDim),
			kNorm: load(prefix+"attn.qknorm.knorm.scale", headDim),
			dim:   dim, heads: heads, kvHeads: kvHeads, headDim: headDim,
		}
	}
	loadFF := func(prefix string, dim, ffn int) ditFF {
		return ditFF{
			gate: tr(load(prefix+"mlp.gate.weight", ffn, dim), ffn, dim),
			up:   tr(load(prefix+"mlp.up.weight", ffn, dim), ffn, dim),
			down: tr(load(prefix+"mlp.down.weight", dim, ffn), dim, ffn),
			dim:  dim, ffn: ffn,
		}
	}

	d.imgInW = tr(load("first.weight", h, cfg.InChannels), h, cfg.InChannels)
	d.imgInB = load("first.bias", h)
	d.tmlp0W = tr(load("tmlp.0.weight", h, cfg.TimestepDim), h, cfg.TimestepDim)
	d.tmlp0B = load("tmlp.0.bias", h)
	d.tmlp2W = tr(load("tmlp.2.weight", h, h), h, h)
	d.tmlp2B = load("tmlp.2.bias", h)
	d.tprojW = tr(load("tproj.1.weight", 6*h, h), 6*h, h)
	d.tprojB = load("tproj.1.bias", 6*h)
	d.txtNorm = load("txtmlp.0.scale", cfg.TextDim)
	d.txtMlp1W = tr(load("txtmlp.1.weight", h, cfg.TextDim), h, cfg.TextDim)
	d.txtMlp1B = load("txtmlp.1.bias", h)
	d.txtMlp3W = tr(load("txtmlp.3.weight", h, h), h, h)
	d.txtMlp3B = load("txtmlp.3.bias", h)
	d.projector = load("txtfusion.projector.weight", cfg.TextLayers)

	textHeadDim := cfg.TextDim / cfg.TextHeads
	for i := 0; i < 2; i++ {
		lp := fmt.Sprintf("txtfusion.layerwise_blocks.%d.", i)
		rp := fmt.Sprintf("txtfusion.refiner_blocks.%d.", i)
		d.fusionLayerwise = append(d.fusionLayerwise, fusionBlock{
			preNorm:  load(lp+"prenorm.scale", cfg.TextDim),
			postNorm: load(lp+"postnorm.scale", cfg.TextDim),
			attn:     loadAttn(lp, cfg.TextDim, cfg.TextHeads, cfg.TextKVHeads, textHeadDim),
			ff:       loadFF(lp, cfg.TextDim, cfg.TextFFN),
		})
		d.fusionRefiner = append(d.fusionRefiner, fusionBlock{
			preNorm:  load(rp+"prenorm.scale", cfg.TextDim),
			postNorm: load(rp+"postnorm.scale", cfg.TextDim),
			attn:     loadAttn(rp, cfg.TextDim, cfg.TextHeads, cfg.TextKVHeads, textHeadDim),
			ff:       loadFF(rp, cfg.TextDim, cfg.TextFFN),
		})
	}

	for i := 0; i < cfg.Layers; i++ {
		p := fmt.Sprintf("blocks.%d.", i)
		d.blocks = append(d.blocks, ditBlock{
			modLin:   load(p+"mod.lin", 6*h),
			preNorm:  load(p+"prenorm.scale", h),
			postNorm: load(p+"postnorm.scale", h),
			attn:     loadAttn(p, h, cfg.Heads, cfg.KVHeads, cfg.HeadDim),
			ff:       loadFF(p, h, cfg.FFN),
		})
		if err != nil {
			return nil, err
		}
	}

	d.lastNorm = load("last.norm.scale", h)
	d.lastModLin = load("last.modulation.lin", 2*h)
	d.lastLinW = tr(load("last.linear.weight", cfg.InChannels, h), cfg.InChannels, h)
	d.lastLB = load("last.linear.bias", cfg.InChannels)
	if err != nil {
		return nil, err
	}
	return d, nil
}

// Forward predicts the flow-matching velocity for packed image tokens.
//
//	img:      [imgSeq, InChannels] packed noisy latents
//	text:     [txtSeq, TextLayers, TextDim] stacked encoder hidden states
//	timestep: flow time in [0,1]
//	pos:      (t,h,w) rope ids for the combined [text|image] sequence
//	txtValid: padding mask for text tokens (nil = all valid)
//
// Returns [imgSeq, InChannels].
func (d *DiT) Forward(img []float32, imgSeq int, text []float32, txtSeq int, timestep float64, pos [][3]int, txtValid []bool) ([]float32, error) {
	cfg := d.cfg
	h := cfg.hidden()
	if len(img) != imgSeq*cfg.InChannels {
		return nil, fmt.Errorf("img length %d != %d*%d", len(img), imgSeq, cfg.InChannels)
	}
	if len(text) != txtSeq*cfg.TextLayers*cfg.TextDim {
		return nil, fmt.Errorf("text length %d != %d*%d*%d", len(text), txtSeq, cfg.TextLayers, cfg.TextDim)
	}
	if len(pos) != txtSeq+imgSeq {
		return nil, fmt.Errorf("pos length %d != %d", len(pos), txtSeq+imgSeq)
	}
	if txtValid == nil {
		txtValid = make([]bool, txtSeq)
		for i := range txtValid {
			txtValid[i] = true
		}
	}

	// Timestep embedding: cos-first sinusoid of t*1000, then MLP.
	temb := d.timestepEmbed(timestep) // [hidden]
	// temb_mod = tproj(gelu_tanh(temb)) + broadcast base for block tables.
	g := make([]float32, h)
	for i, v := range temb {
		g[i] = geluTanh(v)
	}
	tembMod := make([]float32, 6*h)
	matvecHalfInto(tembMod, g, d.tprojW, 6*h)
	addInto(tembMod, d.tprojB)

	// Text fusion: layerwise blocks over the layer axis (each token is a
	// separate "batch" row), projector, refiner blocks over tokens.
	fused := make([]float32, len(text))
	copy(fused, text)
	for bi := range d.fusionLayerwise {
		b := &d.fusionLayerwise[bi]
		// Sequence length = TextLayers; batch = txtSeq tokens.
		var wg sync.WaitGroup
		for tok := 0; tok < txtSeq; tok++ {
			wg.Add(1)
			go func(tok int) {
				defer wg.Done()
				seg := fused[tok*cfg.TextLayers*cfg.TextDim : (tok+1)*cfg.TextLayers*cfg.TextDim]
				fusionBlockForward(b, seg, cfg.TextLayers, nil, cfg.Eps)
			}(tok)
		}
		wg.Wait()
	}
	// Projector: collapse the layer axis per token/dim.
	txt := make([]float32, txtSeq*cfg.TextDim)
	for tok := 0; tok < txtSeq; tok++ {
		base := tok * cfg.TextLayers * cfg.TextDim
		for dim := 0; dim < cfg.TextDim; dim++ {
			var acc float32
			for l := 0; l < cfg.TextLayers; l++ {
				acc += fused[base+l*cfg.TextDim+dim] * d.projector[l]
			}
			txt[tok*cfg.TextDim+dim] = acc
		}
	}
	for bi := range d.fusionRefiner {
		b := &d.fusionRefiner[bi]
		fusionBlockForward(b, txt, txtSeq, txtValid, cfg.Eps)
	}

	// txt_in projection into transformer width.
	txtH := make([]float32, txtSeq*h)
	{
		normed := make([]float32, len(txt))
		rmsNormZeroCenteredInto(normed, txt, d.txtNorm, cfg.TextDim, cfg.Eps)
		matmulHalfInto(txtH, normed, d.txtMlp1W, txtSeq, cfg.TextDim, h)
		addBiasInto(txtH, d.txtMlp1B, txtSeq, h)
		for i, v := range txtH {
			txtH[i] = geluTanh(v)
		}
		tmp := make([]float32, txtSeq*h)
		matmulHalfInto(tmp, txtH, d.txtMlp3W, txtSeq, h, h)
		addBiasInto(tmp, d.txtMlp3B, txtSeq, h)
		txtH = tmp
	}

	// img_in + concat [text | image].
	seq := txtSeq + imgSeq
	x := make([]float32, seq*h)
	copy(x, txtH)
	imgH := x[txtSeq*h:]
	matmulHalfInto(imgH, img, d.imgInW, imgSeq, cfg.InChannels, h)
	addBiasInto(imgH, d.imgInB, imgSeq, h)

	// Combined key-validity: text pads invalid, image all valid.
	keyValid := make([]bool, seq)
	copy(keyValid, txtValid)
	for i := txtSeq; i < seq; i++ {
		keyValid[i] = true
	}

	// Rope tables: per position, interleaved-pair cos/sin per axis segment.
	cosTab, sinTab := d.ropeTables(pos)

	scr := &ditScratch{}
	for bi := range d.blocks {
		d.blockForward(&d.blocks[bi], x, seq, tembMod, cosTab, sinTab, keyValid, scr)
	}

	// Final layer over the image slice only.
	out := make([]float32, imgSeq*cfg.InChannels)
	{
		imgX := x[txtSeq*h:]
		normed := make([]float32, imgSeq*h)
		rmsNormZeroCenteredInto(normed, imgX, d.lastNorm, h, cfg.Eps)
		// modulation = temb + table; chunk -> (scale, shift).
		for t := 0; t < imgSeq; t++ {
			row := normed[t*h : (t+1)*h]
			for i := range row {
				scale := temb[i] + d.lastModLin[i]
				shift := temb[i] + d.lastModLin[h+i]
				row[i] = (1+scale)*row[i] + shift
			}
		}
		matmulHalfInto(out, normed, d.lastLinW, imgSeq, h, cfg.InChannels)
		addBiasInto(out, d.lastLB, imgSeq, cfg.InChannels)
	}
	return out, nil
}

func (d *DiT) timestepEmbed(t float64) []float32 {
	cfg := d.cfg
	h := cfg.hidden()
	half := cfg.TimestepDim / 2
	sin := make([]float32, cfg.TimestepDim)
	for i := 0; i < half; i++ {
		freq := math.Exp(-math.Log(1e4) * float64(i) / float64(half))
		arg := t * 1e3 * freq
		sin[i] = float32(math.Cos(arg))      // cos-first
		sin[half+i] = float32(math.Sin(arg)) // then sin
	}
	h1 := make([]float32, h)
	matvecHalfInto(h1, sin, d.tmlp0W, h)
	addInto(h1, d.tmlp0B)
	for i, v := range h1 {
		h1[i] = geluTanh(v)
	}
	out := make([]float32, h)
	matvecHalfInto(out, h1, d.tmlp2W, h)
	addInto(out, d.tmlp2B)
	return out
}

// ropeTables builds per-position interleaved-pair cos/sin of size HeadDim/2,
// concatenating the (t,h,w) axis segments (diffusers FluxPosEmbed layout).
func (d *DiT) ropeTables(pos [][3]int) (cosTab, sinTab []float32) {
	cfg := d.cfg
	half := cfg.HeadDim / 2
	n := len(pos)
	cosTab = make([]float32, n*half)
	sinTab = make([]float32, n*half)
	for p := 0; p < n; p++ {
		off := 0
		for axis := 0; axis < 3; axis++ {
			ad := cfg.AxesDim[axis]
			for i := 0; i < ad/2; i++ {
				freq := math.Pow(cfg.RopeTheta, -2*float64(i)/float64(ad))
				angle := float64(pos[p][axis]) * freq
				cosTab[p*half+off+i] = float32(math.Cos(angle))
				sinTab[p*half+off+i] = float32(math.Sin(angle))
			}
			off += ad / 2
		}
	}
	return cosTab, sinTab
}

type ditScratch struct {
	normed, modIn, q, k, v, attnOut, proj, gateBuf []float32
	ffGate, ffUp                                   []float32
}

func (s *ditScratch) ensure(seq, h, qDim, kvDim, ffn int) {
	grow := func(b []float32, n int) []float32 {
		if cap(b) < n {
			return make([]float32, n)
		}
		return b[:n]
	}
	s.normed = grow(s.normed, seq*h)
	s.modIn = grow(s.modIn, seq*h)
	s.q = grow(s.q, seq*qDim)
	s.k = grow(s.k, seq*kvDim)
	s.v = grow(s.v, seq*kvDim)
	s.attnOut = grow(s.attnOut, seq*qDim)
	s.proj = grow(s.proj, seq*h)
	s.gateBuf = grow(s.gateBuf, seq*h)
	s.ffGate = grow(s.ffGate, seq*ffn)
	s.ffUp = grow(s.ffUp, seq*ffn)
}

func (d *DiT) blockForward(b *ditBlock, x []float32, seq int, tembMod, cosTab, sinTab []float32, keyValid []bool, s *ditScratch) {
	cfg := d.cfg
	h := cfg.hidden()
	qDim := cfg.Heads * cfg.HeadDim
	kvDim := cfg.KVHeads * cfg.HeadDim
	s.ensure(seq, h, qDim, kvDim, cfg.FFN)

	// mods: (scale1, shift1, gate1, scale2, shift2, gate2), each [h].
	mods := make([]float32, 6*h)
	for i := range mods {
		mods[i] = tembMod[i] + b.modLin[i]
	}
	scale1, shift1, gate1 := mods[0:h], mods[h:2*h], mods[2*h:3*h]
	scale2, shift2, gate2 := mods[3*h:4*h], mods[4*h:5*h], mods[5*h:6*h]

	// --- Attention sublayer ---
	rmsNormZeroCenteredInto(s.normed, x[:seq*h], b.preNorm, h, cfg.Eps)
	modulateInto(s.modIn, s.normed, scale1, shift1, seq, h)
	attnForward(&b.attn, s, s.modIn, seq, cosTab, sinTab, keyValid, false)
	for t := 0; t < seq; t++ {
		for i := 0; i < h; i++ {
			x[t*h+i] += s.proj[t*h+i] * gate1[i]
		}
	}

	// --- MLP sublayer ---
	rmsNormZeroCenteredInto(s.normed, x[:seq*h], b.postNorm, h, cfg.Eps)
	modulateInto(s.modIn, s.normed, scale2, shift2, seq, h)
	ffForward(&b.ff, s, s.modIn, seq)
	for t := 0; t < seq; t++ {
		for i := 0; i < h; i++ {
			x[t*h+i] += s.proj[t*h+i] * gate2[i]
		}
	}
}

// fusionBlockForward runs one Krea text-fusion block in place over
// x [seq, dim] (no rope, no time modulation).
func fusionBlockForward(b *fusionBlock, x []float32, seq int, keyValid []bool, eps float32) {
	dim := b.attn.dim
	s := &ditScratch{}
	s.ensure(seq, dim, b.attn.heads*b.attn.headDim, b.attn.kvHeads*b.attn.headDim, b.ff.ffn)

	rmsNormZeroCenteredInto(s.normed, x[:seq*dim], b.preNorm, dim, eps)
	copy(s.modIn[:seq*dim], s.normed[:seq*dim])
	attnForward(&b.attn, s, s.modIn, seq, nil, nil, keyValid, false)
	for i := 0; i < seq*dim; i++ {
		x[i] += s.proj[i]
	}

	rmsNormZeroCenteredInto(s.normed, x[:seq*dim], b.postNorm, dim, eps)
	copy(s.modIn[:seq*dim], s.normed[:seq*dim])
	ffForward(&b.ff, s, s.modIn, seq)
	for i := 0; i < seq*dim; i++ {
		x[i] += s.proj[i]
	}
}

// attnForward computes Krea attention (GQA, zero-centered per-head QK norm,
// optional interleaved-pair rope, sigmoid output gate) into s.proj.
// cosTab/sinTab nil disables rope. keyValid nil means all keys attendable.
func attnForward(a *ditAttn, s *ditScratch, in []float32, seq int, cosTab, sinTab []float32, keyValid []bool, _ bool) {
	dim := a.dim
	qDim := a.heads * a.headDim
	kvDim := a.kvHeads * a.headDim
	matmulHalfInto(s.q[:seq*qDim], in[:seq*dim], a.wq, seq, dim, qDim)
	matmulHalfInto(s.k[:seq*kvDim], in[:seq*dim], a.wk, seq, dim, kvDim)
	matmulHalfInto(s.v[:seq*kvDim], in[:seq*dim], a.wv, seq, dim, kvDim)
	matmulHalfInto(s.gateBuf[:seq*dim], in[:seq*dim], a.gate, seq, dim, dim)

	applyHeadNormRopeInterleaved(s.q, seq, a.heads, a.headDim, a.qNorm, cosTab, sinTab)
	applyHeadNormRopeInterleaved(s.k, seq, a.kvHeads, a.headDim, a.kNorm, cosTab, sinTab)

	group := a.heads / a.kvHeads
	scale := 1 / float32(math.Sqrt(float64(a.headDim)))
	var wg sync.WaitGroup
	for hIdx := 0; hIdx < a.heads; hIdx++ {
		wg.Add(1)
		go func(hIdx int) {
			defer wg.Done()
			scores := make([]float32, seq)
			kvh := hIdx / group
			for tq := 0; tq < seq; tq++ {
				qv := s.q[(tq*a.heads+hIdx)*a.headDim:][:a.headDim]
				maxScore := float32(math.Inf(-1))
				for tk := 0; tk < seq; tk++ {
					if keyValid != nil && tk < len(keyValid) && !keyValid[tk] {
						scores[tk] = float32(math.Inf(-1))
						continue
					}
					kv := s.k[(tk*a.kvHeads+kvh)*a.headDim:][:a.headDim]
					var dot float32
					for d := 0; d < a.headDim; d++ {
						dot += qv[d] * kv[d]
					}
					dot *= scale
					scores[tk] = dot
					if dot > maxScore {
						maxScore = dot
					}
				}
				outRow := s.attnOut[(tq*a.heads+hIdx)*a.headDim:][:a.headDim]
				for d := range outRow {
					outRow[d] = 0
				}
				var denom float32
				for tk := 0; tk < seq; tk++ {
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
				for tk := 0; tk < seq; tk++ {
					w := scores[tk] * inv
					if w == 0 {
						continue
					}
					vv := s.v[(tk*a.kvHeads+kvh)*a.headDim:][:a.headDim]
					for d := 0; d < a.headDim; d++ {
						outRow[d] += w * vv[d]
					}
				}
			}
		}(hIdx)
	}
	wg.Wait()

	// Sigmoid output gate then wo.
	for i := 0; i < seq*dim; i++ {
		s.attnOut[i] *= sigmoid(s.gateBuf[i])
	}
	matmulHalfInto(s.proj[:seq*dim], s.attnOut[:seq*qDim], a.wo, seq, qDim, dim)
}

func ffForward(f *ditFF, s *ditScratch, in []float32, seq int) {
	matmulHalfInto(s.ffGate[:seq*f.ffn], in[:seq*f.dim], f.gate, seq, f.dim, f.ffn)
	matmulHalfInto(s.ffUp[:seq*f.ffn], in[:seq*f.dim], f.up, seq, f.dim, f.ffn)
	for i := 0; i < seq*f.ffn; i++ {
		g := float64(s.ffGate[i])
		s.ffGate[i] = float32(g/(1+math.Exp(-g))) * s.ffUp[i]
	}
	matmulHalfInto(s.proj[:seq*f.dim], s.ffGate[:seq*f.ffn], f.down, seq, f.ffn, f.dim)
}

// applyHeadNormRopeInterleaved: zero-centered per-head RMSNorm then FLUX-style
// interleaved-pair rotation (pairs are adjacent lanes 2i, 2i+1).
func applyHeadNormRopeInterleaved(qk []float32, seq, heads, headDim int, normW []float32, cosTab, sinTab []float32) {
	half := headDim / 2
	for t := 0; t < seq; t++ {
		for h := 0; h < heads; h++ {
			vec := qk[(t*heads+h)*headDim:][:headDim]
			var ss float64
			for _, v := range vec {
				ss += float64(v) * float64(v)
			}
			inv := float32(1 / math.Sqrt(ss/float64(headDim)+1e-5))
			for i := range vec {
				vec[i] *= inv * (1 + normW[i])
			}
			if cosTab == nil {
				continue
			}
			for i := 0; i < half; i++ {
				c, s := cosTab[t*half+i], sinTab[t*half+i]
				a, b := vec[2*i], vec[2*i+1]
				vec[2*i] = a*c - b*s
				vec[2*i+1] = b*c + a*s
			}
		}
	}
}

func modulateInto(dst, x, scale, shift []float32, seq, h int) {
	for t := 0; t < seq; t++ {
		row := x[t*h : (t+1)*h]
		out := dst[t*h : (t+1)*h]
		for i := range row {
			out[i] = (1+scale[i])*row[i] + shift[i]
		}
	}
}

// rmsNormZeroCenteredInto is RMSNorm with the Krea zero-centered convention:
// multiplier is (1 + weight).
func rmsNormZeroCenteredInto(dst, x, weight []float32, dim int, eps float32) {
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
			out[i] = v * inv * (1 + weight[i])
		}
	}
}

func addInto(dst, b []float32) {
	for i := range dst {
		dst[i] += b[i]
	}
}

func addBiasInto(dst, b []float32, rows, dim int) {
	for r := 0; r < rows; r++ {
		row := dst[r*dim : (r+1)*dim]
		for i := range row {
			row[i] += b[i]
		}
	}
}

func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	x64 := float64(x)
	return float32(0.5 * x64 * (1 + math.Tanh(c*(x64+0.044715*x64*x64*x64))))
}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}
