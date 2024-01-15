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
			MakeDense(Linear, 2, 1),
		},
	}

	net.GradientDescent(x, y, alpha, steps)
	t.Logf("toolkit m0=%v m1=%v b=%v loss=%v", net.Layers[0].W.At(0, 0), net.Layers[0].W.At(0, 1), net.Layers[0].B.At(0, 0), mseLoss2D(x, y, net.Layers[0].W.At(0, 0), net.Layers[0].W.At(0, 1), net.Layers[0].B.At(0, 0)))

	m0, m1, b := gradientDescent2DLinReg(x, y, alpha, steps)
	t.Logf("original m0=%v m1=%v b=%v loss=%v", m0, m1, b, mseLoss2D(x, y, m0, m1, b))

	if math32.Abs(net.Layers[0].W.At(0, 0)-m0) > 0.001 {
		t.Errorf("Disagreement on m0 parameter; got %v, want %v", net.Layers[0].W.At(0, 0), m0)
	}

	if math32.Abs(net.Layers[0].W.At(0, 1)-m1) > 0.001 {
		t.Errorf("Disagreement on m1 parameter; got %v, want %v", net.Layers[0].W.At(0, 1), m1)
	}

	if math32.Abs(net.Layers[0].B.At(0, 0)-b) > 0.001 {
		t.Errorf("Disagreement on b parameter; got %v, want %v", net.Layers[0].B.At(0, 0), b)
	}
}

func generate2DLinRegDataset(m int) (x, y *AF32) {
	r := rand.New(rand.NewSource(12345))

	x = MakeAF32(m, 2)
	y = MakeAF32(m, 1)

	for k := 0; k < m; k++ {
		// Normalization is important --- if I multiply x1 * 1000, the loss is
		// huge and the model blows up with NaNs.
		x0 := r.Float32()
		x1 := r.Float32()
		y0 := 10*x0 + 3*x1 + 30

		// Perturb the point a little bit
		y0 += 0.1*math32.Sin(0.001*x0) + (r.Float32()-0.5)*0.1

		x.Set(k, 0, x0)
		x.Set(k, 1, x1)
		y.Set(k, 0, y0)
	}

	return x, y
}

func mseLoss2D(x, y *AF32, m0, m1, b float32) float32 {
	loss := float32(0)
	for k := 0; k < x.Shape0; k++ {
		pred := m0*x.At(k, 0) + m1*x.At(k, 1) + b
		loss += (pred - y.At(k, 0)) * (pred - y.At(k, 0)) / (2 * float32(x.Shape0))
	}
	return loss
}

func mseLossGradient2D(x, y *AF32, m0, m1, b float32) (gradM0, gradM1, gradB float32) {
	batchSize := x.Shape0
	gradB = float32(0)
	gradM0 = float32(0)
	gradM1 = float32(0)
	for k := 0; k < batchSize; k++ {
		pred := m0*x.At(k, 0) + m1*x.At(k, 1) + b
		gradM0 += (pred - y.At(k, 0)) * x.At(k, 0) / float32(batchSize)
		gradM1 += (pred - y.At(k, 0)) * x.At(k, 1) / float32(batchSize)
		gradB += (pred - y.At(k, 0)) / float32(batchSize)
	}
	return gradM0, gradM1, gradB
}

func gradientDescent2DLinReg(x, y *AF32, learningRate float32, steps int) (m0, m1, b float32) {
	for i := 0; i < steps; i++ {
		gradM0, gradM1, gradB := mseLossGradient2D(x, y, m0, m1, b)
		m0 -= learningRate * gradM0
		m1 -= learningRate * gradM1
		b -= learningRate * gradB
	}
	return m0, m1, b
}
