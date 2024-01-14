package main

import (
	"log"
	"math/rand"

	"github.com/ahmedtd/ml/toolbox"
	"github.com/chewxy/math32"
)

func main() {
	batchSize := 1000
	alpha := float32(0.00001)
	steps := 1000000
	x, y := generateDataset(batchSize)

	net := &toolbox.Network{
		LossFunction: toolbox.SparseCategoricalCrossEntropyFromLogits,
		Layers: []*toolbox.Layer{
			{
				Activation: toolbox.Linear,
				W:          make([]float32, 2*2),
				B:          make([]float32, 2),
				InputSize:  2,
				OutputSize: 2,
			},
		},
	}

	net.GradientDescent(x, y, batchSize, alpha, steps)
	log.Printf("toolbox learned model W=%v B=%v", net.Layers[0].W, net.Layers[0].B)
	log.Printf("toolbox learned decision boundary x2=%v*x1+%v", -net.Layers[0].W[1*2+0]/net.Layers[0].W[1*2+1], -net.Layers[0].B[1]/net.Layers[0].W[1*2+1])

	m := &Model{}
	m.Learn(x, y, batchSize, 0.001, 0.1, steps)
	log.Printf("Learned model W1=%v W2=%v B=%v", m.W1, m.W2, m.B)

	slope := float32(-m.W1 / m.W2)
	intercept := float32(-m.B / m.W2)
	log.Printf("Learned decision boundary x2=%v*x1+%v", slope, intercept)
}

func generateDataset(m int) (x, y []float32) {
	r := rand.New(rand.NewSource(12345))

	x = make([]float32, 2*m)
	y = make([]float32, m)

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
		x[0*m+i] = x1 + 0.0 + float32(0.0*r.NormFloat64())
		x[1*m+i] = x2 + 0.0 + float32(0.0*r.NormFloat64())
		y[i] = y1
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

func (m *Model) loss(x, y []float32, batchSize int, lambda float32) float32 {
	prediction_cost := float32(0)
	regularization_cost := float32(0)

	for i := 0; i < batchSize; i++ {
		pred := sigmoid(m.W1*x[0*batchSize+i] + m.W2*x[1*batchSize+i] + m.B)
		if y[i] == 1.0 {
			prediction_cost += -math32.Log(pred)
		} else {
			prediction_cost += -math32.Log(float32(1) - pred)
		}

		regularization_cost += (m.W1*m.W1 + m.W2*m.W2)
	}

	// The regularization cost is divided by 2n, mostly to make the gradient math simpler.
	return prediction_cost/float32(batchSize) + lambda*regularization_cost/float32(2)/float32(batchSize)
}

func (m *Model) gradient(x, y []float32, batchSize int, lambda float32) (dW1, dW2, dB float32) {
	dW1 = float32(0)
	dW2 = float32(0)
	dB = float32(0)

	for i := 0; i < batchSize; i++ {
		pred := sigmoid(m.W1*x[0*batchSize+i] + m.W2*x[1*batchSize+i] + m.B)
		dW1 += (pred - y[i]) * x[0*batchSize+i]
		dW2 += (pred - y[i]) * x[1*batchSize+i]
		dB += (pred - y[i])
	}

	// Regularize: encourage model parameters to be small.
	dW1 += lambda * m.W1
	dW2 += lambda * m.W2

	dW1 /= float32(batchSize)
	dW2 /= float32(batchSize)
	dB /= float32(batchSize)

	return dW1, dW2, dB
}

func (m *Model) Learn(x, y []float32, batchSize int, learningRate float32, lambda float32, steps int) {
	var dJdW1, dJdW2, dJdb float32
	for i := 0; i < steps; i++ {
		if i < 10 {
			log.Printf("step=%v W1=%v W2=%v B=%v djdw1=%v djdw2=%v djdb=%v loss=%v", i, m.W1, m.W2, m.B, dJdW1, dJdW2, dJdb, m.loss(x, y, batchSize, lambda))
		}

		dJdW1, dJdW2, dJdb = m.gradient(x, y, batchSize, lambda)
		m.W1 -= learningRate * dJdW1
		m.W2 -= learningRate * dJdW2
		m.B -= learningRate * dJdb

		if i%100000 == 0 {
			log.Printf("step=%v W1=%v W2=%v B=%v djdw1=%v djdw2=%v djdb=%v loss=%v", i, m.W1, m.W2, m.B, dJdW1, dJdW2, dJdb, m.loss(x, y, batchSize, lambda))
		}
	}
}
