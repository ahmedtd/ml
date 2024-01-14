package toolbox

import (
	"testing"
)

func BenchmarkApply(b *testing.B) {
	steps := 1000000

	batchSize := b.N
	x, _ := generate2DLinRegDataset(batchSize)

	lay := Layer{
		Activation: Linear,
		W:          make([]float32, 1*2),
		B:          make([]float32, 1),
		InputSize:  2,
		OutputSize: 1,
	}
	z := make([]float32, 1*batchSize)
	a := make([]float32, 1*batchSize)

	b.ResetTimer()

	for s := 0; s < steps; s++ {
		lay.Apply(x, z, a, batchSize)
	}
}

func BenchmarkLinReg(b *testing.B) {
	alpha := float32(0.0001)
	steps := 1000000

	batchSize := b.N
	x, y := generate2DLinRegDataset(batchSize)

	net := &Network{
		LossFunction: MeanSquaredError,
		Layers: []*Layer{
			{
				Activation: Linear,
				W:          make([]float32, 1*2),
				B:          make([]float32, 1),
				InputSize:  2,
				OutputSize: 1,
			},
		},
	}

	b.ResetTimer()
	net.GradientDescent(x, y, batchSize, alpha, steps)
}
