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
func handleNeg(_ *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 1 || inputs[0] == nil {
		return nil, fmt.Errorf("neg: requires 1 input")
	}
	x := inputs[0]
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
func handleInstanceNorm(_ *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
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
