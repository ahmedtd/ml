package toolbox

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestDenseBackpropDjdwSlice(t *testing.T) {

	batchSize := 100
	inputSize := 33
	outputSize := 44

	djdaT := make([]float32, outputSize*batchSize)
	for i := 0; i < outputSize*batchSize; i++ {
		djdaT[i] = 1.0
	}

	dadzT := make([]float32, outputSize*batchSize)
	for i := 0; i < outputSize*batchSize; i++ {
		dadzT[i] = 1.0
	}

	xT := make([]float32, inputSize*batchSize)
	for i := 0; i < inputSize*batchSize; i++ {
		xT[i] = 1.0
	}

	djdw := make([]float32, outputSize*inputSize)

	denseBackpropDjdwSliceKernel(
		batchSize, inputSize,
		djdaT,
		dadzT,
		xT,
		djdw,
		0, 0,
		outputSize*inputSize,
	)

	wantDjdw := make([]float32, outputSize*inputSize)
	for i := 0; i < outputSize*inputSize; i++ {
		wantDjdw[i] = 100.0
	}

	if diff := cmp.Diff(djdw, wantDjdw); diff != "" {
		t.Fatalf("Wrong output; diff (-got +want)\n%s", diff)
	}
}
