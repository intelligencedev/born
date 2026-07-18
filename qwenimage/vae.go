package qwenimage

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// VAEConfig describes the Wan 2.1 VAE decoder (values from the checkpoint
// config; see reference/convert_vae.py output).
type VAEConfig struct {
	BaseDim      int
	ZDim         int
	DimMult      []int
	NumResBlocks int
}

// DefaultVAEConfig matches wan_2.1_vae.safetensors.
func DefaultVAEConfig() VAEConfig {
	return VAEConfig{BaseDim: 96, ZDim: 16, DimMult: []int{1, 2, 4, 4}, NumResBlocks: 2}
}

// conv2d holds a spatial convolution (already reduced from 3D causal form:
// single-frame t2i means only the LAST temporal kernel slice sees data).
type conv2d struct {
	w       []float32 // [out, in, kh, kw]
	b       []float32
	out, in int
	kh, kw  int
	pad     int
}

type vaeResBlock struct {
	norm1, norm2 []float32 // WanRMS gamma [C]
	conv1, conv2 conv2d
	shortcut     *conv2d // 1x1 when in != out
}

type vaeAttn struct {
	norm []float32
	qkv  conv2d // 1x1
	proj conv2d // 1x1
}

type vaeUpBlock struct {
	resnets  []vaeResBlock
	upsample *conv2d // resample conv (after nearest-exact x2); nil on last block
}

// VAE is the pure-Go Wan 2.1 VAE decoder for single-image (t2i) use.
type VAE struct {
	cfg         VAEConfig
	postQuant   conv2d // 1x1 z->z
	convIn      conv2d
	midRes      [2]vaeResBlock
	midAttn     vaeAttn
	ups         []vaeUpBlock
	normOut     []float32
	convOut     conv2d
	LatentsMean []float32 // [ZDim]
	LatentsStd  []float32 // [ZDim]
}

// LoadVAE reads the converted wan-vae-decoder.gguf (diffusers state-dict
// names) via ws, reducing 3D causal convs to 2D at load time.
func LoadVAE(cfg VAEConfig, ws WeightSource) (*VAE, error) {
	v := &VAE{cfg: cfg}

	var err error
	loadConv := func(name string, pad int) conv2d {
		if err != nil {
			return conv2d{}
		}
		w, shape, e := ws.LoadF32(name + ".weight")
		if e != nil {
			err = fmt.Errorf("load %s: %w", name, e)
			return conv2d{}
		}
		b, _, e := ws.LoadF32(name + ".bias")
		if e != nil {
			err = fmt.Errorf("load %s bias: %w", name, e)
			return conv2d{}
		}
		var c conv2d
		switch len(shape) {
		case 5: // (out, in, kt, kh, kw) causal 3D -> take last temporal slice
			out, in, kt, kh, kw := shape[0], shape[1], shape[2], shape[3], shape[4]
			c = conv2d{w: make([]float32, out*in*kh*kw), b: b, out: out, in: in, kh: kh, kw: kw, pad: pad}
			for o := 0; o < out; o++ {
				for i := 0; i < in; i++ {
					src := ((o*in+i)*kt + kt - 1) * kh * kw
					dst := (o*in + i) * kh * kw
					copy(c.w[dst:dst+kh*kw], w[src:src+kh*kw])
				}
			}
		case 4: // plain 2D conv
			c = conv2d{w: w, b: b, out: shape[0], in: shape[1], kh: shape[2], kw: shape[3], pad: pad}
		default:
			err = fmt.Errorf("conv %s: unexpected shape %v", name, shape)
		}
		return c
	}
	loadGamma := func(name string, want int) []float32 {
		if err != nil {
			return nil
		}
		g, shape, e := ws.LoadF32(name + ".gamma")
		if e != nil {
			err = fmt.Errorf("load %s: %w", name, e)
			return nil
		}
		n := 1
		for _, d := range shape {
			n *= d
		}
		if n != want {
			err = fmt.Errorf("%s: %v != %d elems", name, shape, want)
			return nil
		}
		return g
	}
	loadRes := func(prefix string, in, out int) vaeResBlock {
		rb := vaeResBlock{
			norm1: loadGamma(prefix+".norm1", in),
			conv1: loadConv(prefix+".conv1", 1),
			norm2: loadGamma(prefix+".norm2", out),
			conv2: loadConv(prefix+".conv2", 1),
		}
		if in != out {
			sc := loadConv(prefix+".conv_shortcut", 0)
			rb.shortcut = &sc
		}
		return rb
	}

	// dims per WanDecoder3d: [last_mult, reversed mults...] * base.
	m := cfg.DimMult
	dims := []int{cfg.BaseDim * m[len(m)-1]}
	for i := len(m) - 1; i >= 0; i-- {
		dims = append(dims, cfg.BaseDim*m[i])
	}

	v.postQuant = loadConv("post_quant_conv", 0)
	v.convIn = loadConv("decoder.conv_in", 1)
	v.midRes[0] = loadRes("decoder.mid_block.resnets.0", dims[0], dims[0])
	v.midAttn = vaeAttn{
		norm: loadGamma("decoder.mid_block.attentions.0.norm", dims[0]),
		qkv:  loadConv("decoder.mid_block.attentions.0.to_qkv", 0),
		proj: loadConv("decoder.mid_block.attentions.0.proj", 0),
	}
	v.midRes[1] = loadRes("decoder.mid_block.resnets.1", dims[0], dims[0])

	for i := 0; i < len(dims)-1; i++ {
		in, out := dims[i], dims[i+1]
		if i > 0 {
			in = in / 2 // wan 2.1: upsample conv halved the channels
		}
		ub := vaeUpBlock{}
		cur := in
		for j := 0; j <= cfg.NumResBlocks; j++ {
			ub.resnets = append(ub.resnets, loadRes(fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", i, j), cur, out))
			cur = out
		}
		if i != len(dims)-2 {
			up := loadConv(fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0.resample.1", i), 1)
			ub.upsample = &up
		}
		v.ups = append(v.ups, ub)
		if err != nil {
			return nil, err
		}
	}
	v.normOut = loadGamma("decoder.norm_out", dims[len(dims)-1])
	v.convOut = loadConv("decoder.conv_out", 1)
	if err != nil {
		return nil, err
	}
	return v, nil
}

// Decode maps denormalized latents (ZDim, H, W) to RGB floats (3, 8H, 8W)
// in [-1, 1] (caller applies x/2+0.5 postprocessing).
func (v *VAE) Decode(z []float32, hIn, wIn int) ([]float32, int, int, error) {
	cfg := v.cfg
	if len(z) != cfg.ZDim*hIn*wIn {
		return nil, 0, 0, fmt.Errorf("z length %d != %d*%d*%d", len(z), cfg.ZDim, hIn, wIn)
	}
	x, h, w := convApply(&v.postQuant, z, hIn, wIn)
	x, h, w = convApply(&v.convIn, x, h, w)

	x = v.resForward(&v.midRes[0], x, h, w)
	x = v.attnForwardVAE(x, h, w)
	x = v.resForward(&v.midRes[1], x, h, w)

	for i := range v.ups {
		ub := &v.ups[i]
		for j := range ub.resnets {
			x = v.resForward(&ub.resnets[j], x, h, w)
		}
		if ub.upsample != nil {
			x, h, w = upsample2x(x, ub.resnets[len(ub.resnets)-1].conv2.out, h, w)
			x, h, w = convApply(ub.upsample, x, h, w)
		}
	}

	wanRMSNormInPlace(x, v.normOut, h*w)
	siluInPlace(x)
	out, oh, ow := convApply(&v.convOut, x, h, w)
	return out, oh, ow, nil
}

func (v *VAE) resForward(rb *vaeResBlock, x []float32, h, w int) []float32 {
	var short []float32
	if rb.shortcut != nil {
		short, _, _ = convApply(rb.shortcut, x, h, w)
	} else {
		short = make([]float32, len(x))
		copy(short, x)
	}
	t := make([]float32, len(x))
	copy(t, x)
	wanRMSNormInPlace(t, rb.norm1, h*w)
	siluInPlace(t)
	t, _, _ = convApply(&rb.conv1, t, h, w)
	wanRMSNormInPlace(t, rb.norm2, h*w)
	siluInPlace(t)
	t, _, _ = convApply(&rb.conv2, t, h, w)
	for i := range t {
		t[i] += short[i]
	}
	return t
}

// attnForwardVAE: single-head spatial self-attention over H*W tokens.
func (v *VAE) attnForwardVAE(x []float32, h, w int) []float32 {
	c := v.midAttn.qkv.in
	n := h * w
	t := make([]float32, len(x))
	copy(t, x)
	wanRMSNormInPlace(t, v.midAttn.norm, n)
	qkv, _, _ := convApply(&v.midAttn.qkv, t, h, w) // (3c, h, w)
	q, k, vv := qkv[:c*n], qkv[c*n:2*c*n], qkv[2*c*n:]

	// channel-major -> token vectors on the fly; scale 1/sqrt(c).
	scale := 1 / float32(math.Sqrt(float64(c)))
	out := make([]float32, c*n)
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for start := 0; start < n; start += chunk {
		end := start + chunk
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			scores := make([]float32, n)
			for tq := start; tq < end; tq++ {
				maxScore := float32(math.Inf(-1))
				for tk := 0; tk < n; tk++ {
					var dot float32
					for ch := 0; ch < c; ch++ {
						dot += q[ch*n+tq] * k[ch*n+tk]
					}
					dot *= scale
					scores[tk] = dot
					if dot > maxScore {
						maxScore = dot
					}
				}
				var denom float32
				for tk := 0; tk < n; tk++ {
					e := float32(math.Exp(float64(scores[tk] - maxScore)))
					scores[tk] = e
					denom += e
				}
				inv := 1 / denom
				for tk := 0; tk < n; tk++ {
					sw := scores[tk] * inv
					if sw == 0 {
						continue
					}
					for ch := 0; ch < c; ch++ {
						out[ch*n+tq] += sw * vv[ch*n+tk]
					}
				}
			}
		}(start, end)
	}
	wg.Wait()

	proj, _, _ := convApply(&v.midAttn.proj, out, h, w)
	for i := range proj {
		proj[i] += x[i]
	}
	return proj
}

// convApply runs a 2D convolution (channel-major input [C,H,W]) with same
// padding pad and stride 1, parallel over output channels.
func convApply(c *conv2d, x []float32, h, w int) ([]float32, int, int) {
	out := make([]float32, c.out*h*w)
	var wg sync.WaitGroup
	for o := 0; o < c.out; o++ {
		wg.Add(1)
		go func(o int) {
			defer wg.Done()
			dst := out[o*h*w : (o+1)*h*w]
			for i := range dst {
				dst[i] = c.b[o]
			}
			for in := 0; in < c.in; in++ {
				src := x[in*h*w : (in+1)*h*w]
				wBase := ((o*c.in + in) * c.kh) * c.kw
				for ky := 0; ky < c.kh; ky++ {
					for kx := 0; kx < c.kw; kx++ {
						wv := c.w[wBase+ky*c.kw+kx]
						if wv == 0 {
							continue
						}
						dy := ky - c.pad
						dx := kx - c.pad
						yStart, yEnd := 0, h
						if dy < 0 {
							yStart = -dy
						}
						if dy > 0 {
							yEnd = h - dy
						}
						for y := yStart; y < yEnd; y++ {
							sy := y + dy
							xStart, xEnd := 0, w
							if dx < 0 {
								xStart = -dx
							}
							if dx > 0 {
								xEnd = w - dx
							}
							srow := src[sy*w:]
							drow := dst[y*w:]
							for xx := xStart; xx < xEnd; xx++ {
								drow[xx] += wv * srow[xx+dx]
							}
						}
					}
				}
			}
		}(o)
	}
	wg.Wait()
	return out, h, w
}

// upsample2x performs nearest-exact 2x spatial upsampling (pixel duplication).
func upsample2x(x []float32, c, h, w int) ([]float32, int, int) {
	oh, ow := h*2, w*2
	out := make([]float32, c*oh*ow)
	for ch := 0; ch < c; ch++ {
		src := x[ch*h*w:]
		dst := out[ch*oh*ow:]
		for y := 0; y < oh; y++ {
			sy := y / 2
			for xx := 0; xx < ow; xx++ {
				dst[y*ow+xx] = src[sy*w+xx/2]
			}
		}
	}
	return out, oh, ow
}

// wanRMSNormInPlace: L2-normalize across channels per spatial position, then
// multiply by sqrt(C)*gamma (F.normalize eps 1e-12).
func wanRMSNormInPlace(x, gamma []float32, spatial int) {
	c := len(x) / spatial
	scale := float32(math.Sqrt(float64(c)))
	for p := 0; p < spatial; p++ {
		var ss float64
		for ch := 0; ch < c; ch++ {
			v := float64(x[ch*spatial+p])
			ss += v * v
		}
		norm := float32(math.Sqrt(ss))
		if norm < 1e-12 {
			norm = 1e-12
		}
		inv := scale / norm
		for ch := 0; ch < c; ch++ {
			x[ch*spatial+p] *= inv * gamma[ch]
		}
	}
}

func siluInPlace(x []float32) {
	for i, v := range x {
		x[i] = v * sigmoid(v)
	}
}
