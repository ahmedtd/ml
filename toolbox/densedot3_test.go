//go:build goexperiment.simd && amd64

package toolbox

import (
	"math/rand"
	"strconv"
	"testing"
)

func BenchmarkDenseDot3(b *testing.B) {
	b.Run("impl=naive", func(b *testing.B) {
		for i := 8; i < 16; i++ {
			b.Run("size="+strconv.Itoa(2<<i), func(b *testing.B) {
				x := make([]float32, 2<<i)
				y := make([]float32, 2<<i)
				z := make([]float32, 2<<i)
				for i := range 2 << i {
					x[i] = rand.Float32()
					y[i] = rand.Float32()
					z[i] = rand.Float32()
				}
				for b.Loop() {
					_ = denseDot3Naive(x, y, z)
				}
			})
		}
	})
	b.Run("impl=simd", func(b *testing.B) {
		for i := 8; i < 16; i++ {
			b.Run("size="+strconv.Itoa(2<<i), func(b *testing.B) {
				x := make([]float32, 2<<i)
				y := make([]float32, 2<<i)
				z := make([]float32, 2<<i)
				for i := range 2 << i {
					x[i] = rand.Float32()
					y[i] = rand.Float32()
					z[i] = rand.Float32()
				}
				for b.Loop() {
					_ = denseDot3SIMD(x, y, z)
				}
			})
		}
	})
}
