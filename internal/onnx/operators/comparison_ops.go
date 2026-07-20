//go:build !wasm

package operators

import (
	"fmt"

	"github.com/intelligencedev/born/internal/tensor"
)

// registerComparisonOps adds comparison operators to the registry.
func (r *Registry) registerComparisonOps() {
	r.Register("Equal", handleEqual)
	r.Register("Greater", handleGreater)
	r.Register("GreaterOrEqual", handleGreaterOrEqual)
	r.Register("Less", handleLess)
	r.Register("LessOrEqual", handleLessOrEqual)
}

func handleEqual(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("equal requires 2 inputs, got %d", len(inputs))
	}
	result := comparisonBackend(ctx, inputs).Equal(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleGreater(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("greater requires 2 inputs, got %d", len(inputs))
	}
	result := comparisonBackend(ctx, inputs).Greater(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleGreaterOrEqual(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("greaterOrEqual requires 2 inputs, got %d", len(inputs))
	}
	result := comparisonBackend(ctx, inputs).GreaterEqual(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleLess(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("less requires 2 inputs, got %d", len(inputs))
	}
	result := comparisonBackend(ctx, inputs).Lower(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func handleLessOrEqual(ctx *Context, _ *Node, inputs []*tensor.RawTensor) ([]*tensor.RawTensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("lessOrEqual requires 2 inputs, got %d", len(inputs))
	}
	result := comparisonBackend(ctx, inputs).LowerEqual(inputs[0], inputs[1])
	return []*tensor.RawTensor{result}, nil
}

func comparisonBackend(ctx *Context, inputs []*tensor.RawTensor) tensor.Backend {
	if inputs[0].DType() == tensor.Int64 && inputs[1].DType() == tensor.Int64 {
		return integerMathBackend
	}
	return ctx.Backend
}
