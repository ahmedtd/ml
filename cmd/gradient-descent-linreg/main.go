package main

import (
	"flag"
	"log"
	"math/rand"

	"github.com/ahmedtd/ml/toolbox"
	"github.com/chewxy/math32"
)

func main() {
	flag.Parse()

	alpha := float32(0.0001)
	steps := 1000000

	batchSize := 1000
	x, y := generate1DLinRegDataset(batchSize)

	net := &toolbox.Network{
		LossFunction: toolbox.MeanSquaredError,
		Layers: []*toolbox.Layer{
			toolbox.MakeDense(toolbox.Linear, 1, 1),
		},
	}

	net.GradientDescent(x, y, alpha, steps)
	log.Printf("toolkit m=%v b=%v loss=%v", net.Layers[0].W.At(0, 0), net.Layers[0].B.At(0, 0), net.Loss(x, y))
	log.Printf("toolkit loss=%v", net.Loss(x, y))

	m, b := gradientDescentLinReg(x, y, alpha, steps, float32(0.0), float32(0.0))
	log.Printf("original m=%v b=%v loss=%v", m, b, lossFn(x, y, m, b))
}

func generate1DLinRegDataset(m int) (x, y *toolbox.AF32) {
	r := rand.New(rand.NewSource(12345))

	x = toolbox.MakeAF32(m, 1)
	y = toolbox.MakeAF32(m, 1)

	for i := 0; i < m; i++ {
		// Normalization is important --- if I multiply x1 * 1000, the loss is
		// huge and the model blows up with NaNs.
		x1 := r.Float32()
		y1 := 10*x1 + 30

		// Perturb the point a little bit
		y1 += 0.1*math32.Sin(0.001*x1) + (r.Float32()-0.5)*10

		x.Set(i, 0, x1)
		y.Set(i, 0, y1)
	}

	return x, y
}

func lossFn(x, y *toolbox.AF32, m, b float32) float32 {
	loss := float32(0)
	for i := 0; i < x.Shape0; i++ {
		pred := m*x.At(i, 0) + b
		loss += (pred - y.At(i, 0)) * (pred - y.At(i, 0)) / (2 * float32(x.Shape0))
	}
	return loss
}

func gradientFn(x, y *toolbox.AF32, m, b float32) (gradM, gradB float32) {
	gradB = float32(0)
	gradM = float32(0)
	for i := 0; i < x.Shape0; i++ {
		pred := m*x.At(i, 0) + b
		gradM += (pred - y.At(i, 0)) * x.At(i, 0) / float32(x.Shape0)
		gradB += (pred - y.At(i, 0)) / float32(x.Shape0)
	}
	return gradM, gradB
}

func gradientDescentLinReg(x, y *toolbox.AF32, learningRate float32, steps int, initM, initB float32) (m, b float32) {
	m = initM
	b = initB
	for i := 0; i < steps; i++ {
		gradM, gradB := gradientFn(x, y, m, b)
		m = m - learningRate*gradM
		b = b - learningRate*gradB
	}
	return m, b
}
