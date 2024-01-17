package toolbox

import (
	"testing"
)

func BenchmarkApply(b *testing.B) {
	batchSize := b.N
	x, _ := generate2DLinRegDataset(batchSize, 10, 5, 30)

	lay := MakeDense(Linear, 2, 1)
	z := MakeAF32(batchSize, 1)
	a := MakeAF32(batchSize, 1)

	b.ResetTimer()
	lay.Apply(x, z, a, 0, batchSize)
}

func BenchmarkLinReg(b *testing.B) {
	alpha := float32(0.0001)
	steps := 1000

	batchSize := b.N
	x, y := generate2DLinRegDataset(batchSize, 10, 5, 30)

	net := &Network{
		LossFunction: MeanSquaredError,
		Layers: []*Layer{
			MakeDense(Linear, 2, 1),
		},
	}

	b.ResetTimer()
	net.GradientDescent(x, y, alpha, steps)
}
