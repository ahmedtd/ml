//go:build goexperiment.simd && amd64

package toolbox

import (
	"math/rand"
	"strconv"
	"testing"
)

func BenchmarkDenseDot2(b *testing.B) {
	b.Run("impl=naive", func(b *testing.B) {
		for i := 8; i < 16; i++ {
			b.Run("size="+strconv.Itoa(2<<i), func(b *testing.B) {
				x := make([]float32, 2<<i)
				y := make([]float32, 2<<i)
				for i := range 2 << i {
					x[i] = rand.Float32()
					y[i] = rand.Float32()
				}
				for b.Loop() {
					_ = denseDot2Naive(x, y)
				}
			})
		}
	})
	b.Run("impl=simd", func(b *testing.B) {
		for i := 8; i < 16; i++ {
			b.Run("size="+strconv.Itoa(2<<i), func(b *testing.B) {
				x := make([]float32, 2<<i)
				y := make([]float32, 2<<i)
				for i := range 2 << i {
					x[i] = rand.Float32()
					y[i] = rand.Float32()
				}
				for b.Loop() {
					_ = denseDot2SIMD(x, y)
				}
			})
		}
	})
}
