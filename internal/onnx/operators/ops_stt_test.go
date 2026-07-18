//go:build !wasm

package operators

import (
	"testing"

	"github.com/intelligencedev/born/internal/tensor"
)

func TestRangeInt64AndFloat32(t *testing.T) {
	out := stExec(t, &Node{OpType: "Range"},
		stI64(t, tensor.Shape{}, []int64{2}), stI64(t, tensor.Shape{}, []int64{10}), stI64(t, tensor.Shape{}, []int64{3}))
	if out.DType() != tensor.Int64 {
		t.Fatalf("dtype = %s", out.DType())
	}
	got := out.AsInt64()
	want := []int64{2, 5, 8}
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("got %v want %v", got, want)
	}
	outF := stExec(t, &Node{OpType: "Range"},
		stF32(t, tensor.Shape{}, []float32{0}), stF32(t, tensor.Shape{}, []float32{1}), stF32(t, tensor.Shape{}, []float32{0.4}))
	stAssertClose(t, outF.AsFloat32(), []float32{0, 0.4, 0.8})
}

func TestNeg(t *testing.T) {
	out := stExec(t, &Node{OpType: "Neg"}, stF32(t, tensor.Shape{4}, []float32{1, -2, 0, 3.5}))
	stAssertClose(t, out.AsFloat32(), []float32{-1, 2, 0, -3.5})
}

// numpy: x=[[1,2,3,4],[10,20,30,40]] per-channel mean/var; scale=[2,1], bias=[1,0], eps=0
// ch0: mean 2.5 var 1.25 -> (x-2.5)/sqrt(1.25)*2+1 = [-1.6833, -0.7889, 0.7889, 1.6833]+... let's use eps=1e-5 values from numpy below.
func TestInstanceNormalization(t *testing.T) {
	x := stF32(t, tensor.Shape{1, 2, 4}, []float32{1, 2, 3, 4, 10, 20, 30, 40})
	scale := stF32(t, tensor.Shape{2}, []float32{2, 1})
	bias := stF32(t, tensor.Shape{2}, []float32{1, 0})
	out := stExec(t, &Node{OpType: "InstanceNormalization",
		Attributes: []Attribute{{Name: "epsilon", F: 1e-5}}}, x, scale, bias)
	stAssertShape(t, out, tensor.Shape{1, 2, 4})
	stAssertClose(t, out.AsFloat32(), []float32{
		-1.68328, 0.10557, 1.89443, 3.68328,
		-1.34164, -0.44721, 0.44721, 1.34164,
	})
}

// Trilu: upper (default) keeps elements on/above the k-th diagonal.
func TestTrilu(t *testing.T) {
	x := stF32(t, tensor.Shape{3, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	up := stExec(t, &Node{OpType: "Trilu"}, x)
	stAssertClose(t, up.AsFloat32(), []float32{1, 2, 3, 0, 5, 6, 0, 0, 9})
	lo := stExec(t, &Node{OpType: "Trilu", Attributes: []Attribute{{Name: "upper", I: 0}}},
		stF32(t, tensor.Shape{3, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	stAssertClose(t, lo.AsFloat32(), []float32{1, 0, 0, 4, 5, 0, 7, 8, 9})
	// k=1 upper: shift diagonal right by one
	k := stI64(t, tensor.Shape{}, []int64{1})
	up1 := stExec(t, &Node{OpType: "Trilu"}, x, k)
	stAssertClose(t, up1.AsFloat32(), []float32{0, 2, 3, 0, 0, 6, 0, 0, 0})
}
