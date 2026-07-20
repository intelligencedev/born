//go:build !wasm

package operators

import (
	"fmt"
	"math"

	"github.com/intelligencedev/born/internal/tensor"
)

// registerSTTOps registers operators required by speech-to-text encoder graphs
// (Moonshine/Whisper family) beyond the base + Supertonic sets.
func (r *Registry) registerSTTOps() {
	r.Register("Range", handleRange)
	r.Register("Neg", handleNeg)
	r.Register("InstanceNormalization", handleInstanceNorm)
	r.Register("Trilu", handleTrilu)
}

// handleRange implements ONNX Range(start, limit, delta) for int64 and float32
// scalars: a 1-D tensor of max(ceil((limit-start)/delta), 0) elements.
func handleRange(_ *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return nil, fmt.Errorf("range: requires start, limit, delta")
	}
	switch inputs[0].DType() {
	case tensor.Int64:
		start, limit, delta := inputs[0].AsInt64()[0], inputs[1].AsInt64()[0], inputs[2].AsInt64()[0]
		if delta == 0 {
			return nil, fmt.Errorf("range: zero delta")
		}
		n := (limit - start + delta - sign64(delta)) / delta
		if n < 0 {
			n = 0
		}
		out, err := tensor.NewRaw(tensor.Shape{int(n)}, tensor.Int64, inputs[0].Device())
		if err != nil {
			return nil, err
		}
		d := out.AsInt64()
		v := start
		for i := range d {
			d[i] = v
			v += delta
		}
		return []*tensor.RawTensor{out}, nil
	case tensor.Float32:
		start := float64(inputs[0].AsFloat32()[0])
		limit := float64(inputs[1].AsFloat32()[0])
		delta := float64(inputs[2].AsFloat32()[0])
		if delta == 0 {
			return nil, fmt.Errorf("range: zero delta")
		}
		n := int(math.Ceil((limit - start) / delta))
		if n < 0 {
			n = 0
		}
		out, err := tensor.NewRaw(tensor.Shape{n}, tensor.Float32, inputs[0].Device())
		if err != nil {
			return nil, err
		}
		d := out.AsFloat32()
		for i := range d {
			d[i] = float32(start + float64(i)*delta)
		}
		return []*tensor.RawTensor{out}, nil
	default:
		return nil, fmt.Errorf("range: unsupported dtype %s", inputs[0].DType())
	}
}

func sign64(v int64) int64 {
	if v < 0 {
		return -1
	}
	return 1
}

// handleNeg implements elementwise negation for float32 and int64.
func handleNeg(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 1 || inputs[0] == nil {
		return nil, fmt.Errorf("neg: requires 1 input")
	}
	x := inputs[0]
	if x.DType() == tensor.Float32 {
		return []*tensor.RawTensor{ctx.Backend.MulScalar(x, float32(-1))}, nil
	}
	out, err := tensor.NewRaw(x.Shape(), x.DType(), x.Device())
	if err != nil {
		return nil, err
	}
	switch x.DType() {
	case tensor.Float32:
		in, o := x.AsFloat32(), out.AsFloat32()
		for i, v := range in {
			o[i] = -v
		}
	case tensor.Int64:
		in, o := x.AsInt64(), out.AsInt64()
		for i, v := range in {
			o[i] = -v
		}
	default:
		return nil, fmt.Errorf("neg: unsupported dtype %s", x.DType())
	}
	return []*tensor.RawTensor{out}, nil
}

// handleInstanceNorm implements ONNX InstanceNormalization for float32
// [N, C, spatial...]: per-(N,C) mean/variance over spatial dims, then
// y = scale[c]*(x-mean)/sqrt(var+eps) + bias[c].
func handleInstanceNorm(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return nil, fmt.Errorf("instancenorm: requires x, scale, bias")
	}
	x := inputs[0]
	if x.DType() != tensor.Float32 {
		return nil, fmt.Errorf("instancenorm: only float32 supported")
	}
	shape := x.Shape()
	if len(shape) < 3 {
		return nil, fmt.Errorf("instancenorm: expected >=3D input, got %v", shape)
	}
	eps := float64(GetAttrFloat(node, "epsilon", 1e-5))
	if backend, ok := ctx.Backend.(tensor.STTBackend); ok {
		out, err := backend.InstanceNormalization(x, inputs[1], inputs[2], float32(eps))
		if err != nil {
			return nil, fmt.Errorf("instancenorm: backend: %w", err)
		}
		return []*tensor.RawTensor{out}, nil
	}
	n, c := shape[0], shape[1]
	spatial := 1
	for _, d := range shape[2:] {
		spatial *= d
	}
	scale, bias := inputs[1].AsFloat32(), inputs[2].AsFloat32()
	if len(scale) != c || len(bias) != c {
		return nil, fmt.Errorf("instancenorm: scale/bias length %d/%d, want %d", len(scale), len(bias), c)
	}
	out, err := tensor.NewRaw(shape, tensor.Float32, x.Device())
	if err != nil {
		return nil, err
	}
	xd, od := x.AsFloat32(), out.AsFloat32()
	for ni := 0; ni < n; ni++ {
		for ci := 0; ci < c; ci++ {
			seg := xd[(ni*c+ci)*spatial:][:spatial]
			var mean float64
			for _, v := range seg {
				mean += float64(v)
			}
			mean /= float64(spatial)
			var varSum float64
			for _, v := range seg {
				d := float64(v) - mean
				varSum += d * d
			}
			inv := 1.0 / math.Sqrt(varSum/float64(spatial)+eps)
			s, b := float64(scale[ci]), float64(bias[ci])
			oseg := od[(ni*c+ci)*spatial:][:spatial]
			for i, v := range seg {
				oseg[i] = float32((float64(v)-mean)*inv*s + b)
			}
		}
	}
	return []*tensor.RawTensor{out}, nil
}

// handleTrilu implements ONNX Trilu for float32 [..., R, C]: upper=1 (default)
// zeroes elements below the k-th diagonal, upper=0 zeroes above it. k comes
// from the optional second input (int64 scalar, default 0).
func handleTrilu(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) < 1 || inputs[0] == nil {
		return nil, fmt.Errorf("trilu: requires input")
	}
	x := inputs[0]
	if x.DType() != tensor.Float32 {
		return nil, fmt.Errorf("trilu: only float32 supported, got %s", x.DType())
	}
	shape := x.Shape()
	if len(shape) < 2 {
		return nil, fmt.Errorf("trilu: expected >=2D, got %v", shape)
	}
	var k int64
	if len(inputs) >= 2 && inputs[1] != nil && inputs[1].NumElements() > 0 {
		k = inputs[1].AsInt64()[0]
	}
	upper := GetAttrInt(node, "upper", 1) != 0
	rows, cols := shape[len(shape)-2], shape[len(shape)-1]
	batch := 1
	for _, d := range shape[:len(shape)-2] {
		batch *= d
	}
	mask, err := tensor.NewRaw(shape, tensor.Float32, tensor.CPU)
	if err != nil {
		return nil, err
	}
	maskData := mask.AsFloat32()
	for b := 0; b < batch; b++ {
		base := b * rows * cols
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				keep := int64(c-r) >= k
				if !upper {
					keep = int64(c-r) <= k
				}
				if keep {
					maskData[base+r*cols+c] = 1
				}
			}
		}
	}
	return []*tensor.RawTensor{ctx.Backend.Mul(x, mask)}, nil
}
