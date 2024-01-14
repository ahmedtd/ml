package toolbox

import (
	"math/rand"
	"testing"

	"github.com/chewxy/math32"
)

func TestAgreesWithHandcodedLinreg(t *testing.T) {
	alpha := float32(0.0001)
	steps := 1000000

	batchSize := 1000
	x, y := generate1DLinRegDataset(batchSize)

	net := &Network{
		LossFunction: MeanSquaredError,
		Layers: []*Layer{
			{
				Activation: Linear,
				W:          make([]float32, 1*1),
				B:          make([]float32, 1),
				InputSize:  1,
				OutputSize: 1,
			},
		},
	}

	net.GradientDescent(x, y, batchSize, alpha, steps)
	t.Logf("toolkit m=%v b=%v loss=%v", net.Layers[0].W[0], net.Layers[0].B[0], lossFn(x, y, net.Layers[0].W[0], net.Layers[0].B[0]))

	m, b := gradientDescentLinReg(x, y, alpha, steps, float32(0.0), float32(0.0))
	t.Logf("original m=%v b=%v loss=%v", m, b, lossFn(x, y, m, b))

	if math32.Abs(net.Layers[0].W[0]-m) > 0.001 {
		t.Errorf("Disagreement on m parameter; got %v, want %v", net.Layers[0].W[0], m)
	}

	if math32.Abs(net.Layers[0].B[0]-b) > 0.001 {
		t.Errorf("Disagreement on b parameter; got %v, want %v", net.Layers[0].B[0], b)
	}
}

func generate1DLinRegDataset(m int) (x, y []float32) {
	r := rand.New(rand.NewSource(12345))

	x = make([]float32, m)
	y = make([]float32, m)

	for i := 0; i < m; i++ {
		// Normalization is important --- if I multiply x1 * 1000, the loss is
		// huge and the model blows up with NaNs.
		x1 := r.Float32()
		y1 := 10*x1 + 30

		// Perturb the point a little bit
		y1 += 0.1*math32.Sin(0.001*x1) + (r.Float32()-0.5)*10

		x[0*m+i] = x1
		y[i] = y1
	}

	return x, y
}

func lossFn(x, y []float32, m, b float32) float32 {
	loss := float32(0)
	for i := range x {
		pred := m*x[i] + b
		loss += (pred - y[i]) * (pred - y[i]) / (2 * float32(len(x)))
	}
	return loss
}

func gradientFn(x, y []float32, m, b float32) (gradM, gradB float32) {
	gradB = float32(0)
	gradM = float32(0)
	for i := range x {
		pred := m*x[i] + b
		gradM += (pred - y[i]) * x[i] / float32(len(x))
		gradB += (pred - y[i]) / float32(len(x))
	}
	return gradM, gradB
}

func gradientDescentLinReg(x, y []float32, learningRate float32, steps int, initM, initB float32) (m, b float32) {
	m = initM
	b = initB
	for i := 0; i < steps; i++ {
		gradM, gradB := gradientFn(x, y, m, b)
		m = m - learningRate*gradM
		b = b - learningRate*gradB
	}
	return m, b
}
