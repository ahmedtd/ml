//go:build goexperiment.simd && amd64

package toolbox

import (
	"simd/archsimd"
)

func denseDot2Naive(x []float32, y []float32) float32 {
	if len(x) != len(y) {
		panic("mismatched length")
	}
	var sum float32
	for i := range len(x) {
		sum += x[i] * y[i]
	}
	return sum
}

// https://go.dev/play/p/NY5rJYPoJcl
func denseDot2SIMD(x []float32, y []float32) float32 {
	// if len(x) != len(y) {
	// 	panic("mismatched length")
	// }
	// var sum float32
	// for i := range len(x) {
	// 	sum += x[i] * y[i]
	// }
	// return sum

	var (
		s0, s1, s2, s3 archsimd.Float32x8
	)

	// Writing anything slice indexing related in constant can reduce the bound checks.
	// Our bound-check elimination pass is clever at reasoning constants, but struggles
	// at reasoning expressions with variables.
	for len(x) >= 32 && len(y) >= 32 {
		x3 := archsimd.LoadFloat32x8Slice(x[24:])
		x2 := archsimd.LoadFloat32x8Slice(x[16:])
		x1 := archsimd.LoadFloat32x8Slice(x[8:])
		x0 := archsimd.LoadFloat32x8Slice(x[:])
		x = x[32:]
		y3 := archsimd.LoadFloat32x8Slice(y[24:])
		y2 := archsimd.LoadFloat32x8Slice(y[16:])
		y1 := archsimd.LoadFloat32x8Slice(y[8:])
		y0 := archsimd.LoadFloat32x8Slice(y[:])
		y = y[32:]

		s0 = x0.MulAdd(y0, s0)
		s1 = x1.MulAdd(y1, s1)
		s2 = x2.MulAdd(y2, s2)
		s3 = x3.MulAdd(y3, s3)
	}

	// Reduce to one value
	s0 = s0.Add(s1).Add(s2.Add(s3))
	low, high := s0.GetLo(), s0.GetHi()
	sum4 := low.Add(high)
	sum2 := sum4.AddPairs(sum4)
	sum1 := sum2.AddPairs(sum2)
	sum1Slice := make([]float32, 4)
	sum1.StoreSlice(sum1Slice)
	sum := sum1Slice[0]

	// Handle the tail.
	if len(x) == len(y) {
		// Again remove on unnecessary bound check.
		// Our bound-check elimination pass is also clever at reasoning ==.
		for i := range len(x) {
			sum += x[i] * y[i]
		}
	}

	return sum
}

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

// z (input/output)
func reluActivation(z []float32) {
	var z0, z1, z2, z3, zeros archsimd.Float32x8
	for len(z) >= 32 {
		z3 = archsimd.LoadFloat32x8Slice(z[24:])
		z2 = archsimd.LoadFloat32x8Slice(z[16:])
		z1 = archsimd.LoadFloat32x8Slice(z[8:])
		z0 = archsimd.LoadFloat32x8Slice(z[:])

		z3.Max(zeros).StoreSlice(z[24:])
		z2.Max(zeros).StoreSlice(z[16:])
		z1.Max(zeros).StoreSlice(z[8:])
		z0.Max(zeros).StoreSlice(z[:])

		z = z[32:]
	}

	// Handle tail of less than 32 but more than 8 elements.
	for len(z) >= 8 {
		z0 = archsimd.LoadFloat32x8Slice(z)
		z0.Max(zeros).StoreSlice(z)
		z = z[8:]
	}

	// Handle final tail of less than 8 elements
	if len(z) > 0 {
		z0 = archsimd.LoadFloat32x8SlicePart(z)
		z0.Max(zeros).StoreSlicePart(z)
	}
}
