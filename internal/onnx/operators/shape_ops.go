//go:build !wasm

package operators

import (
	"fmt"
	"os"
	"sort"

	"github.com/intelligencedev/born/internal/tensor"
)

// registerShapeOps adds shape manipulation operators to the registry.
func (r *Registry) registerShapeOps() {
	r.Register("Reshape", handleReshape)
	r.Register("Transpose", handleTranspose)
	r.Register("Squeeze", handleSqueeze)
	r.Register("Unsqueeze", handleUnsqueeze)
	r.Register("Concat", handleConcat)
	r.Register("Split", handleSplit)
	r.Register("Slice", handleSlice)
	r.Register("Gather", handleGather)
	r.Register("Flatten", handleFlatten)
	r.Register("Expand", handleExpand)
}

func handleReshape(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("reshape requires 2 inputs (data, shape), got %d", len(inputs))
	}

	// Get target shape from second input. Per ONNX Reshape semantics
	// (allowzero=0 default), a 0 entry means "copy the corresponding dim from
	// the input shape"; only -1 is inferred.
	shapeData := inputs[1].AsInt64()
	inShape := inputs[0].Shape()
	newShape := make(tensor.Shape, len(shapeData))
	for i, v := range shapeData {
		if v == 0 && i < len(inShape) {
			newShape[i] = inShape[i]
		} else {
			newShape[i] = int(v)
		}
	}

	if inputs[0].DType() == tensor.Float32 {
		resolvedShape, err := resolveReshape(newShape, inputs[0].NumElements())
		if err != nil {
			return nil, fmt.Errorf("reshape: %w", err)
		}
		return []*tensor.RawTensor{ctx.Backend.Reshape(inputs[0], resolvedShape)}, nil
	}
	result, err := tensor.Reshape(inputs[0], newShape)
	if err != nil {
		return nil, fmt.Errorf("reshape: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleTranspose(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("transpose requires 1 input, got %d", len(inputs))
	}

	perm := GetAttrInts(node, "perm")
	var axes []int
	if len(perm) > 0 {
		axes = make([]int, len(perm))
		for i, v := range perm {
			axes[i] = int(v)
		}
	}

	if inputs[0].DType() == tensor.Float32 {
		return []*tensor.RawTensor{ctx.Backend.Transpose(inputs[0], axes...)}, nil
	}
	result, err := tensor.TransposeAxes(inputs[0], axes...)
	if err != nil {
		return nil, fmt.Errorf("transpose: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleSqueeze(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("squeeze requires at least 1 input, got %d", len(inputs))
	}

	// ONNX 13+: axes is second input
	var axes []int
	if len(inputs) >= 2 && inputs[1] != nil {
		axesData := inputs[1].AsInt64()
		axes = make([]int, len(axesData))
		for i, v := range axesData {
			axes[i] = int(v)
		}
	} else {
		// Fall back to attribute
		axesAttr := GetAttrInts(node, "axes")
		if len(axesAttr) > 0 {
			axes = make([]int, len(axesAttr))
			for i, v := range axesAttr {
				axes[i] = int(v)
			}
		}
	}

	if inputs[0].DType() == tensor.Float32 && len(axes) > 0 {
		normalized := normalizeAxes(axes, len(inputs[0].Shape()))
		sort.Sort(sort.Reverse(sort.IntSlice(normalized)))
		result := inputs[0]
		for _, axis := range normalized {
			result = ctx.Backend.Squeeze(result, axis)
		}
		return []*tensor.RawTensor{result}, nil
	}
	result, err := tensor.Squeeze(inputs[0], axes...)
	if err != nil {
		return nil, fmt.Errorf("squeeze: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleUnsqueeze(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("unsqueeze requires at least 1 input, got %d", len(inputs))
	}

	// ONNX 13+: axes is second input
	var axes []int
	if len(inputs) >= 2 && inputs[1] != nil {
		axesData := inputs[1].AsInt64()
		axes = make([]int, len(axesData))
		for i, v := range axesData {
			axes[i] = int(v)
		}
	} else {
		axesAttr := GetAttrInts(node, "axes")
		if len(axesAttr) > 0 {
			axes = make([]int, len(axesAttr))
			for i, v := range axesAttr {
				axes[i] = int(v)
			}
		}
	}

	if len(axes) == 0 {
		return nil, fmt.Errorf("unsqueeze requires axes")
	}
	if inputs[0].DType() == tensor.Float32 {
		outputRank := len(inputs[0].Shape()) + len(axes)
		normalized := normalizeAxes(axes, outputRank)
		sort.Ints(normalized)
		result := inputs[0]
		for _, axis := range normalized {
			result = ctx.Backend.Unsqueeze(result, axis)
		}
		return []*tensor.RawTensor{result}, nil
	}

	result, err := tensor.Unsqueeze(inputs[0], axes...)
	if err != nil {
		return nil, fmt.Errorf("unsqueeze: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleConcat(ctx *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("concat requires at least 1 input")
	}

	axis := int(GetAttrInt(node, "axis", 0))
	allFloat32 := true
	for _, input := range inputs {
		allFloat32 = allFloat32 && input != nil && input.DType() == tensor.Float32
	}
	if allFloat32 {
		return []*tensor.RawTensor{ctx.Backend.Cat(inputs, axis)}, nil
	}

	result, err := tensor.Concat(inputs, axis)
	if err != nil {
		return nil, fmt.Errorf("concat: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleSplit(_ *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("split requires at least 1 input, got %d", len(inputs))
	}

	axis := int(GetAttrInt(node, "axis", 0))

	// Get split sizes
	var splitSizes []int
	if len(inputs) >= 2 && inputs[1] != nil {
		// ONNX 13+: split sizes from input
		sizesData := inputs[1].AsInt64()
		splitSizes = make([]int, len(sizesData))
		for i, v := range sizesData {
			splitSizes[i] = int(v)
		}
	} else {
		// Fall back to attribute
		sizesAttr := GetAttrInts(node, "split")
		if len(sizesAttr) > 0 {
			splitSizes = make([]int, len(sizesAttr))
			for i, v := range sizesAttr {
				splitSizes[i] = int(v)
			}
		}
	}

	results, err := tensor.Split(inputs[0], axis, splitSizes)
	if err != nil {
		return nil, fmt.Errorf("split: %w", err)
	}
	return results, nil
}

func handleSlice(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) < 3 {
		return nil, fmt.Errorf("slice requires at least 3 inputs (data, starts, ends), got %d", len(inputs))
	}

	starts := inputs[1].AsInt64()
	ends := inputs[2].AsInt64()

	var axes, steps []int64
	if len(inputs) >= 4 && inputs[3] != nil {
		axes = inputs[3].AsInt64()
	}
	if len(inputs) >= 5 && inputs[4] != nil {
		steps = inputs[4].AsInt64()
	}

	if bornSliceDebug {
		fmt.Printf("SLICE in=%v starts=%v ends=%v axes=%v steps=%v\n", inputs[0].Shape(), starts, ends, axes, steps)
	}
	if inputs[0].DType() == tensor.Float32 {
		if backend, ok := ctx.Backend.(tensor.STTBackend); ok {
			result, err := backend.Slice(inputs[0], starts, ends, axes, steps)
			if err != nil {
				return nil, fmt.Errorf("slice: backend: %w", err)
			}
			return []*tensor.RawTensor{result}, nil
		}
	}

	result, err := tensor.Slice(inputs[0], starts, ends, axes, steps)
	if err != nil {
		return nil, fmt.Errorf("slice: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleGather(_ *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("gather requires 2 inputs (data, indices), got %d", len(inputs))
	}

	axis := int(GetAttrInt(node, "axis", 0))

	result, err := tensor.Gather(inputs[0], inputs[1], axis)
	if err != nil {
		return nil, fmt.Errorf("gather: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleFlatten(_ *Context, node *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("flatten requires 1 input, got %d", len(inputs))
	}

	axis := int(GetAttrInt(node, "axis", 1))

	result, err := tensor.Flatten(inputs[0], axis)
	if err != nil {
		return nil, fmt.Errorf("flatten: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func handleExpand(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("expand requires 2 inputs (input, shape), got %d", len(inputs))
	}

	// Get target shape from second input
	shapeData := inputs[1].AsInt64()
	targetShape := make(tensor.Shape, len(shapeData))
	for i, v := range shapeData {
		targetShape[i] = int(v)
	}

	// ONNX Expand broadcasts the input against the target shape (bidirectional),
	// so a target dim of 1 keeps the input's (possibly larger) dim.
	outShape, err := onnxBroadcastShape(inputs[0].Shape(), targetShape)
	if err != nil {
		return nil, fmt.Errorf("expand: %w", err)
	}

	if inputs[0].DType() == tensor.Float32 {
		return []*tensor.RawTensor{ctx.Backend.Expand(inputs[0], outShape)}, nil
	}
	result, err := tensor.Expand(inputs[0], outShape)
	if err != nil {
		return nil, fmt.Errorf("expand: %w", err)
	}
	return []*tensor.RawTensor{result}, nil
}

func normalizeAxes(axes []int, rank int) []int {
	normalized := make([]int, len(axes))
	for index, axis := range axes {
		if axis < 0 {
			axis += rank
		}
		normalized[index] = axis
	}
	return normalized
}

func resolveReshape(shape tensor.Shape, elements int) (tensor.Shape, error) {
	resolved := shape.Clone()
	inferred := -1
	knownProduct := 1
	for index, dimension := range resolved {
		switch {
		case dimension == -1:
			if inferred >= 0 {
				return nil, fmt.Errorf("multiple inferred dimensions")
			}
			inferred = index
		case dimension < 0:
			return nil, fmt.Errorf("invalid dimension %d", dimension)
		default:
			knownProduct *= dimension
		}
	}
	if inferred >= 0 {
		if knownProduct == 0 || elements%knownProduct != 0 {
			return nil, fmt.Errorf("cannot infer shape %v for %d elements", shape, elements)
		}
		resolved[inferred] = elements / knownProduct
	}
	return resolved, nil
}

//nolint:gochecknoglobals // debug flag
var bornSliceDebug = os.Getenv("BORN_SLICE_DEBUG") != ""
