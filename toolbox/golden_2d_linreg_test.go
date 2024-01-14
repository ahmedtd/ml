package toolbox

import (
	"math/rand"
	"testing"

	"github.com/chewxy/math32"
)

func TestAgreesWithGolden2DLinreg(t *testing.T) {
	alpha := float32(0.0001)
	steps := 1000000

	batchSize := 1000
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

	net.GradientDescent(x, y, batchSize, alpha, steps)
	t.Logf("toolkit m0=%v m1=%v b=%v loss=%v", net.Layers[0].W[0], net.Layers[0].W[1], net.Layers[0].B[0], mseLoss2D(x, y, batchSize, net.Layers[0].W[0], net.Layers[0].W[1], net.Layers[0].B[0]))

	m0, m1, b := gradientDescent2DLinReg(x, y, batchSize, alpha, steps)
	t.Logf("original m0=%v m1=%v b=%v loss=%v", m0, m1, b, mseLoss2D(x, y, batchSize, m0, m1, b))

	if math32.Abs(net.Layers[0].W[0]-m0) > 0.001 {
		t.Errorf("Disagreement on m0 parameter; got %v, want %v", net.Layers[0].W[0], m0)
	}

	if math32.Abs(net.Layers[0].W[1]-m1) > 0.001 {
		t.Errorf("Disagreement on m1 parameter; got %v, want %v", net.Layers[0].W[1], m1)
	}

	if math32.Abs(net.Layers[0].B[0]-b) > 0.001 {
		t.Errorf("Disagreement on b parameter; got %v, want %v", net.Layers[0].B[0], b)
	}
}

func generate2DLinRegDataset(m int) (x, y []float32) {
	r := rand.New(rand.NewSource(12345))

	x = make([]float32, 2*m)
	y = make([]float32, m)

	for k := 0; k < m; k++ {
		// Normalization is important --- if I multiply x1 * 1000, the loss is
		// huge and the model blows up with NaNs.
		x0 := r.Float32()
		x1 := r.Float32()
		y0 := 10*x0 + 3*x1 + 30

		// Perturb the point a little bit
		y0 += 0.1*math32.Sin(0.001*x0) + (r.Float32()-0.5)*0.1

		x[k*2+0] = x0
		x[k*2+1] = x1
		y[k] = y0
	}

	return x, y
}

func mseLoss2D(x, y []float32, batchSize int, m0, m1, b float32) float32 {
	loss := float32(0)
	for k := 0; k < batchSize; k++ {
		pred := m0*x[k*2+0] + m1*x[k*2+1] + b
		loss += (pred - y[k]) * (pred - y[k]) / (2 * float32(batchSize))
	}
	return loss
}

func mseLossGradient2D(x, y []float32, batchSize int, m0, m1, b float32) (gradM0, gradM1, gradB float32) {
	gradB = float32(0)
	gradM0 = float32(0)
	gradM1 = float32(0)
	for k := 0; k < batchSize; k++ {
		pred := m0*x[k*2+0] + m1*x[k*2+1] + b
		gradM0 += (pred - y[k]) * x[k*2+0] / float32(batchSize)
		gradM1 += (pred - y[k]) * x[k*2+1] / float32(batchSize)
		gradB += (pred - y[k]) / float32(batchSize)
	}
	return gradM0, gradM1, gradB
}

func gradientDescent2DLinReg(x, y []float32, batchSize int, learningRate float32, steps int) (m0, m1, b float32) {
	for i := 0; i < steps; i++ {
		gradM0, gradM1, gradB := mseLossGradient2D(x, y, batchSize, m0, m1, b)
		m0 -= learningRate * gradM0
		m1 -= learningRate * gradM1
		b -= learningRate * gradB
	}
	return m0, m1, b
}
