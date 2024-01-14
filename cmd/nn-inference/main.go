package main

import (
	"fmt"

	"github.com/chewxy/math32"
)

func main() {

}

type Vector struct {
	V    []float32
	Size int
}

type Layer struct {
	W []float32
	B []float32

	InputSize  int
	OutputSize int
}

func sigmoid(z float32) float32 {
	return float32(1) / (float32(1) + float32(math32.Exp(-z)))
}

func (l *Layer) apply(in, out *Vector) error {
	if len(l.W) != l.InputSize*l.OutputSize {
		return fmt.Errorf("inconsistency: len(l.W) != l.OutputSize * l.InputSize")
	}

	if len(l.B) != l.OutputSize {
		return fmt.Errorf("inconsistency: len(l.B) != l.OutputSize")
	}

	if in.Size != l.InputSize {
		return fmt.Errorf("inconsistency: in.Size != l.InputSize")
	}

	if out.Size != l.OutputSize {
		return fmt.Errorf("inconsistency: out.Size != l.OutputSize")
	}

	for i := 0; i < l.OutputSize; i++ {
		accum := float32(0.0)
		for j := 0; j < l.InputSize; j++ {
			accum += l.W[i*l.InputSize+j] * in.V[j]
		}
		accum += l.B[i]
		out.V[i] = sigmoid(accum)
	}

	return nil
}
