package main

import (
	"flag"
	"log"
)

func main() {
	flag.Parse()

	x := []float32{1.0, 2.0, 3.0}
	y := []float32{3.0, 4.0, 5.0}

	m, b := gradientDescentLinReg(x, y, 0.0001, 1000000, float32(0.0), float32(0.0))
	log.Printf("m=%v b=%v loss=%v", m, b, lossFn(x, y, m, b))
}

func lossFn(x, y []float32, m, b float32) float32 {
	loss := float32(0)
	for i := range x {
		pred := m*x[i] + b
		loss += (pred - y[i]) * (pred - y[i])
	}
	loss /= 2 * float32(len(x))
	return loss
}

func gradientFn(x, y []float32, m, b float32) (gradM, gradB float32) {
	gradB = float32(0)
	gradM = float32(0)
	for i := range x {
		pred := m*x[i] + b
		gradM += (pred - y[i]) * x[i]
		gradB += (pred - y[i])
	}
	gradM /= float32(len(x))
	gradB /= float32(len(x))
	return gradM, gradB
}

func gradientDescentLinReg(x, y []float32, learningRate float32, steps int, initM, initB float32) (m, b float32) {
	m = initM
	b = initB
	for i := 0; i < steps; i++ {
		gradM, gradB := gradientFn(x, y, m, b)
		m = m - learningRate*gradM
		b = b - learningRate*gradB
		if i%1000 == 0 {
			log.Printf("step=%v m=%v b=%v gradM=%v gradB=%v loss=%v", i, m, b, gradM, gradB, lossFn(x, y, m, b))
		}
	}
	return m, b
}
