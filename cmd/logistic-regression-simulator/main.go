package main

import (
	"log"
	"math/rand"

	"github.com/ahmedtd/ml/toolbox"
	"github.com/chewxy/math32"
)

func main() {
	batchSize := 1000
	alpha := float32(0.001)
	steps := 1000000
	x, y := generateDataset(batchSize)

	num0s := 0
	num1s := 0
	for k := 0; k < batchSize; k++ {
		if y.At(k, 0) == 1 {
			num1s++
		} else {
			num0s++
		}
	}
	log.Printf("original data set has %d 1s and %d 0s", num1s, num0s)

	net := &toolbox.Network{
		LossFunction: toolbox.SparseCategoricalCrossEntropyFromLogits,
		Layers: []*toolbox.Layer{
			toolbox.MakeDense(toolbox.Linear, 2, 2),
		},
	}

	net.GradientDescent(x, y, alpha, steps)
	log.Printf("toolbox learned model W=%v B=%v loss=%v", net.Layers[0].W, net.Layers[0].B, net.Loss(x, y))
	log.Printf("toolbox learned decision boundary x1=%v*x0+%v", -net.Layers[0].W.At(1, 0)/net.Layers[0].W.At(1, 1), -net.Layers[0].B.At(1, 0)/net.Layers[0].W.At(1, 1))

	toolboxPredictions := net.Apply(x)
	toolboxNumMispredictions := 0
	for k := 0; k < batchSize; k++ {
		prediction := float32(0.0)
		if toolboxPredictions.At(k, 1) > toolboxPredictions.At(k, 0) {
			prediction = 1.0
		}

		if prediction != y.At(k, 0) {
			toolboxNumMispredictions++
			// log.Printf("toolbox mispredicted k=%d x0=%v x1=%v pred=%v truth=%v", k, x.At(k, 0), x.At(k, 1), prediction, y.At(k, 0))
		}
	}
	log.Printf("toolbox had %d mispredictions (%v%%)", toolboxNumMispredictions, float32(toolboxNumMispredictions)/float32(batchSize)*float32(100))

	m := &Model{}
	m.Learn(x, y, alpha, 0.0, steps)
	log.Printf("Learned model W1=%v W2=%v B=%v", m.W1, m.W2, m.B)

	slope := float32(-m.W1 / m.W2)
	intercept := float32(-m.B / m.W2)
	log.Printf("Learned decision boundary x2=%v*x1+%v", slope, intercept)

	handPredictions := m.apply(x)
	handNumMispredictions := 0
	for k := 0; k < batchSize; k++ {
		if handPredictions.At(k, 0) != y.At(k, 0) {
			handNumMispredictions++
			// log.Printf("hand mispredicted k=%d x0=%v x1=%v pred=%v truth=%v", k, x.At(k, 0), x.At(k, 1), handPredictions.At(k, 0), y.At(k, 0))
		}
	}
	log.Printf("hand had %d mispredictions (%v%%)", handNumMispredictions, float32(handNumMispredictions)/float32(batchSize)*float32(100))

}

func generateDataset(m int) (x, y *toolbox.AF32) {
	r := rand.New(rand.NewSource(12345))

	x = toolbox.MakeAF32(m, 2)
	y = toolbox.MakeAF32(m, 1)

	for i := 0; i < m; i++ {
		// Generate a point and classify it according to the "true"
		// distribution.
		x1 := (r.Float32())
		x2 := (r.Float32())
		y1 := float32(0.0)
		if x2 > 1.0*x1+0.0 {
			y1 = 1.0
		}

		// Perturb the point a little bit with noise and constant bias.
		x.Set(i, 0, x1+0.0+float32(0.0*r.NormFloat64()))
		x.Set(i, 1, x2+0.0+float32(0.0*r.NormFloat64()))
		y.Set(i, 0, y1)
	}

	return x, y
}

type Model struct {
	W1, W2 float32
	B      float32
}

func sigmoid(z float32) float32 {
	return float32(1) / (float32(1) + float32(math32.Exp(-z)))
}

func (m *Model) apply(x *toolbox.AF32) *toolbox.AF32 {
	batchSize := x.Shape0

	pred := toolbox.MakeAF32(batchSize, 1)

	for k := 0; k < batchSize; k++ {
		if sigmoid(m.W1*x.At(k, 0)+m.W2*x.At(k, 1)+m.B) > 0.5 {
			pred.Set(k, 0, 1)
		} else {
			pred.Set(k, 0, 0)
		}
	}
	return pred
}

func (m *Model) loss(x, y *toolbox.AF32, lambda float32) float32 {
	batchSize := x.Shape0

	prediction_cost := float32(0)
	regularization_cost := float32(0)

	for i := 0; i < batchSize; i++ {
		pred := sigmoid(m.W1*x.At(i, 0) + m.W2*x.At(i, 1) + m.B)
		if y.At(i, 0) == 1.0 {
			prediction_cost += -math32.Log(pred)
		} else {
			prediction_cost += -math32.Log(float32(1) - pred)
		}

		regularization_cost += (m.W1*m.W1 + m.W2*m.W2)
	}

	// The regularization cost is divided by 2n, mostly to make the gradient math simpler.
	return prediction_cost/float32(batchSize) + lambda*regularization_cost/float32(2)/float32(batchSize)
}

func (m *Model) gradient(x, y *toolbox.AF32, lambda float32) (dW1, dW2, dB float32) {
	batchSize := x.Shape0

	dW1 = float32(0)
	dW2 = float32(0)
	dB = float32(0)

	for i := 0; i < batchSize; i++ {
		pred := sigmoid(m.W1*x.At(i, 0) + m.W2*x.At(i, 1) + m.B)
		dW1 += (pred - y.At(i, 0)) * x.At(i, 0)
		dW2 += (pred - y.At(i, 0)) * x.At(i, 1)
		dB += (pred - y.At(i, 0))
	}

	// Regularize: encourage model parameters to be small.
	dW1 += lambda * m.W1
	dW2 += lambda * m.W2

	dW1 /= float32(batchSize)
	dW2 /= float32(batchSize)
	dB /= float32(batchSize)

	return dW1, dW2, dB
}

func (m *Model) Learn(x, y *toolbox.AF32, learningRate float32, lambda float32, steps int) {
	var dJdW1, dJdW2, dJdb float32
	for i := 0; i < steps; i++ {
		dJdW1, dJdW2, dJdb = m.gradient(x, y, lambda)
		m.W1 -= learningRate * dJdW1
		m.W2 -= learningRate * dJdW2
		m.B -= learningRate * dJdb

		if i%100000 == 0 {
			log.Printf("step=%v W1=%v W2=%v B=%v djdw1=%v djdw2=%v djdb=%v loss=%v", i, m.W1, m.W2, m.B, dJdW1, dJdW2, dJdb, m.loss(x, y, lambda))
		}
	}
}
