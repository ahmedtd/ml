//go:build !goexperiment.simd || !amd64

package toolbox

func denseDot2SIMD(x []float32, y []float32) float32 {
	if len(x) != len(y) {
		panic("mismatched length")
	}
	var sum float32
	for i := range len(x) {
		sum += x[i] * y[i]
	}
	return sum
}

func denseDot3SIMD(x, y, z []float32) float32 {
	if len(x) != len(y) || len(x) != len(z) {
		panic("all input slices must have the same length")
	}
	var sum float32
	for i := range len(x) {
		sum += x[i] * y[i] * z[i]
	}
	return sum
}

// z (input/output)
func reluActivation(z []float32) {
	for i := range len(z) {
		if z[i] < 0.0 {
			z[i] = 0.0
		}
	}
}
