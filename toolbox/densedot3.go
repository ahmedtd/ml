//go:build goexperiment.simd && amd64

package toolbox

import "simd/archsimd"

func denseDot3Naive(x, y, z []float32) float32 {
	if len(x) != len(y) || len(x) != len(z) {
		panic("all input slices must have the same length")
	}
	var sum float32
	for i := range len(x) {
		sum += x[i] * y[i] * z[i]
	}
	return sum
}

func denseDot3SIMD(x, y, z []float32) float32 {
	var a archsimd.Float32x8
	i := 0
	for ; i < len(x)-8; i += 8 { // this idiom is friendly to bounds check elimination
		xv := archsimd.LoadFloat32x8Slice(x[i : i+8])
		yv := archsimd.LoadFloat32x8Slice(y[i : i+8])
		zv := archsimd.LoadFloat32x8Slice(z[i : i+8])
		a = xv.Mul(yv).MulAdd(zv, a)
	}
	xv := archsimd.LoadFloat32x8SlicePart(x[i:])
	yv := archsimd.LoadFloat32x8SlicePart(y[i:])
	zv := archsimd.LoadFloat32x8SlicePart(z[i:])
	a = xv.Mul(yv).MulAdd(zv, a)
	a = a.AddPairsGrouped(a) // 01234567                AP 01234567                -> 0+1 2+3 _ _ 4+5 6+7 _ _
	a = a.AddPairsGrouped(a) // 0+1 2+3 _ _ 4+5 6+7 _ _ AP 0+1 2+3 _ _ 4+5 6+7 _ _ -> 0+1+2+3 _ _ _ 4+5+6+7 _ _ _
	b := a.GetLo().Add(a.GetHi())
	return b.GetElem(0)
}
