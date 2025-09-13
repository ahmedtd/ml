//go:build goexperiment.simd && amd64

package toolbox

import "simd"

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
		s0, s1, s2, s3 simd.Float32x8
	)

	// Writing anything slice indexing related in constant can reduce the bound checks.
	// Our bound-check elimination pass is clever at reasoning constants, but struggles
	// at reasoning expressions with variables.
	for len(x) >= 32 && len(y) >= 32 {
		x3 := simd.LoadFloat32x8Slice(x[24:])
		x2 := simd.LoadFloat32x8Slice(x[16:])
		x1 := simd.LoadFloat32x8Slice(x[8:])
		x0 := simd.LoadFloat32x8Slice(x[:])
		x = x[32:]
		y3 := simd.LoadFloat32x8Slice(y[24:])
		y2 := simd.LoadFloat32x8Slice(y[16:])
		y1 := simd.LoadFloat32x8Slice(y[8:])
		y0 := simd.LoadFloat32x8Slice(y[:])
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
