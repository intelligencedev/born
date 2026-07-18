package qwenimage

import (
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/intelligencedev/born/internal/gguf"
)

// Model file names expected inside the model directory.
const (
	ditFile       = "Krea-2-Turbo-Q4_K_M.gguf"
	encoderFile   = "Qwen3VL-4B-Instruct-Q8_0.gguf"
	vaeFile       = "wan-vae-decoder.gguf" // produced by reference/convert_vae.py
	tokenizerFile = "tokenizer/tokenizer.json"
)

// Krea2 conditioning constants (see reference/ORACLE.md).
const (
	krea2Prefix       = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
	krea2Suffix       = "<|im_end|>\n<|im_start|>assistant\n"
	krea2PrefixTokens = 34
	krea2SuffixTokens = 5
	krea2MaxSeq       = 512
	padTokenID        = 151643 // <|endoftext|>
)

var selectLayers = []int{2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}

// Options controls one generation.
type Options struct {
	Width, Height int
	Steps         int
	Seed          uint64
	OnStep        func(step, total int, d time.Duration)
}

// Pipeline generates images with Krea 2 Turbo fully in-process.
// Memory strategy: the text encoder (f32) is loaded, run, and released
// before the DiT (f16) is loaded; peak resident is the DiT's ~24 GB.
type Pipeline struct {
	dir string
	tok *Tokenizer
}

// New validates the model directory layout and loads the tokenizer.
func New(modelDir string) (*Pipeline, error) {
	for _, f := range []string{ditFile, encoderFile, vaeFile, tokenizerFile} {
		if _, err := os.Stat(filepath.Join(modelDir, f)); err != nil {
			return nil, fmt.Errorf("model dir %s: missing %s (expected layout: %s, %s, %s, %s): %w",
				modelDir, f, ditFile, encoderFile, vaeFile, tokenizerFile, err)
		}
	}
	tok, err := LoadTokenizer(filepath.Join(modelDir, "tokenizer"))
	if err != nil {
		return nil, err
	}
	return &Pipeline{dir: modelDir, tok: tok}, nil
}

func openGGUF(path string) (*gguf.TensorConverter, func(), error) {
	f, err := gguf.ParseFile(path)
	if err != nil {
		return nil, nil, err
	}
	conv, err := gguf.NewTensorConverter(f)
	if err != nil {
		return nil, nil, err
	}
	return conv, func() { _ = conv.Close() }, nil
}

// buildInputs assembles the fixed Krea 2 layout
// [prefix | prompt | PAD | suffix] with its validity mask.
func (p *Pipeline) buildInputs(prompt string) (ids []int32, valid []bool) {
	prefix := p.tok.Encode(krea2Prefix)
	body := p.tok.Encode(prompt)
	maxBody := krea2MaxSeq - krea2SuffixTokens // prompt space before padding
	if len(body) > maxBody {
		body = body[:maxBody]
	}
	suffix := p.tok.Encode(krea2Suffix)
	total := krea2PrefixTokens + krea2MaxSeq
	ids = make([]int32, 0, total)
	valid = make([]bool, 0, total)
	push := func(t []int32, v bool) {
		for _, id := range t {
			ids = append(ids, id)
			valid = append(valid, v)
		}
	}
	push(prefix, true)
	push(body, true)
	for len(ids) < total-len(suffix) {
		ids = append(ids, padTokenID)
		valid = append(valid, false)
	}
	push(suffix, true)
	return ids, valid
}

// Generate produces an image for prompt. Cancellation is honored between
// diffusion steps.
func (p *Pipeline) Generate(ctx context.Context, prompt string, opt Options) (image.Image, error) {
	if opt.Width <= 0 {
		opt.Width = 512
	}
	if opt.Height <= 0 {
		opt.Height = 512
	}
	if opt.Steps <= 0 {
		opt.Steps = 8
	}
	if opt.Width%16 != 0 || opt.Height%16 != 0 {
		return nil, fmt.Errorf("width/height must be multiples of 16, got %dx%d", opt.Width, opt.Height)
	}

	// --- Text conditioning (encoder freed before the DiT loads) ---
	ids, valid := p.buildInputs(prompt)
	text, txtValid, err := p.encodeText(ids, valid)
	if err != nil {
		return nil, err
	}
	runtime.GC()
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// --- Diffusion ---
	gridH := opt.Height / 16
	gridW := opt.Width / 16
	imgSeq := gridH * gridW
	txtSeq := krea2MaxSeq

	conv, done, err := openGGUF(filepath.Join(p.dir, ditFile))
	if err != nil {
		return nil, err
	}
	dit, err := LoadDiT(DefaultDiTConfig(), ggufWeightsSource{conv})
	done()
	if err != nil {
		return nil, err
	}

	pos := make([][3]int, txtSeq+imgSeq)
	for i := 0; i < imgSeq; i++ {
		pos[txtSeq+i] = [3]int{0, i / gridW, i % gridW}
	}

	latents := NoiseLatents(opt.Seed, imgSeq*64)
	sigmas := Sigmas(opt.Steps)
	for step := 0; step < opt.Steps; step++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		start := time.Now()
		vel, err := dit.Forward(latents, imgSeq, text, txtSeq, sigmas[step], pos, txtValid)
		if err != nil {
			return nil, err
		}
		EulerStep(latents, vel, sigmas[step], sigmas[step+1])
		if opt.OnStep != nil {
			opt.OnStep(step+1, opt.Steps, time.Since(start))
		}
	}
	dit = nil
	runtime.GC()

	// --- Decode ---
	return p.decode(latents, gridH, gridW)
}

// encodeText loads the Qwen3-VL encoder, runs the fixed layout, stacks the
// select layers into (512, 12, 2560) and returns the post-drop mask.
func (p *Pipeline) encodeText(ids []int32, valid []bool) ([]float32, []bool, error) {
	conv, done, err := openGGUF(filepath.Join(p.dir, encoderFile))
	if err != nil {
		return nil, nil, err
	}
	defer done()
	te, err := LoadTextEncoder(DefaultTextEncoderConfig(), ggufWeightsSource{conv})
	if err != nil {
		return nil, nil, err
	}
	layers, err := te.Forward(ids, valid, selectLayers)
	if err != nil {
		return nil, nil, err
	}
	cfg := DefaultTextEncoderConfig()
	seq := len(ids) - krea2PrefixTokens
	text := make([]float32, seq*len(selectLayers)*cfg.Hidden)
	for tok := 0; tok < seq; tok++ {
		src := (tok + krea2PrefixTokens) * cfg.Hidden
		for li := range selectLayers {
			copy(text[(tok*len(selectLayers)+li)*cfg.Hidden:], layers[li][src:src+cfg.Hidden])
		}
	}
	return text, valid[krea2PrefixTokens:], nil
}

// decode unpacks (imgSeq, 64) latents, denormalizes and runs the VAE.
func (p *Pipeline) decode(latents []float32, gridH, gridW int) (image.Image, error) {
	conv, done, err := openGGUF(filepath.Join(p.dir, vaeFile))
	if err != nil {
		return nil, err
	}
	defer done()
	vae, err := LoadVAE(DefaultVAEConfig(), ggufWeightsSource{conv})
	if err != nil {
		return nil, err
	}
	mean, _ := loadVAEStats(p.dir, "latents_mean")
	std, _ := loadVAEStats(p.dir, "latents_std")

	// Unpack: token (y,x) holds a (16, 2, 2) sub-block at c*4 + py*2 + px.
	lh, lw := gridH*2, gridW*2
	z := make([]float32, 16*lh*lw)
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			tokBase := (y*gridW + x) * 64
			for c := 0; c < 16; c++ {
				for py := 0; py < 2; py++ {
					for px := 0; px < 2; px++ {
						z[c*lh*lw+(2*y+py)*lw+(2*x+px)] = latents[tokBase+c*4+py*2+px]
					}
				}
			}
		}
	}
	// Denormalize per channel: z * std + mean.
	for c := 0; c < 16; c++ {
		for i := 0; i < lh*lw; i++ {
			z[c*lh*lw+i] = z[c*lh*lw+i]*std[c] + mean[c]
		}
	}

	rgb, oh, ow, err := vae.Decode(z, lh, lw)
	if err != nil {
		return nil, err
	}
	img := image.NewRGBA(image.Rect(0, 0, ow, oh))
	for y := 0; y < oh; y++ {
		for x := 0; x < ow; x++ {
			off := img.PixOffset(x, y)
			for ch := 0; ch < 3; ch++ {
				v := rgb[ch*oh*ow+y*ow+x]/2 + 0.5
				if v < 0 {
					v = 0
				} else if v > 1 {
					v = 1
				}
				img.Pix[off+ch] = uint8(v*255 + 0.5)
			}
			img.Pix[off+3] = 255
		}
	}
	return img, nil
}

// loadVAEStats reads latents_mean/std from the converted VAE gguf metadata
// fixture written by convert_vae.py alongside the weights.
func loadVAEStats(dir, which string) ([]float32, error) {
	// Wan 2.1 constants (from the checkpoint config; recorded in
	// reference/fixtures/vae.latents_{mean,std}).
	means := []float32{
		-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
		0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
	}
	stds := []float32{
		2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
		3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
	}
	_ = dir
	if which == "latents_mean" {
		return means, nil
	}
	return stds, nil
}

// ggufWeightsSource adapts TensorConverter to WeightSource (non-test twin of
// the test helper).
type ggufWeightsSource struct {
	conv *gguf.TensorConverter
}

func (g ggufWeightsSource) LoadF32(name string) ([]float32, []int, error) {
	return g.conv.Convert(name)
}
