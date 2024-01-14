package toolbox

import (
	"fmt"

	"github.com/chewxy/math32"
)

// Verify bounds check elimination with
//
//   go build -gcflags="-d=ssa/check_bce" ./toolbox/`

type Array struct {
	V     []float32
	Shape []int
}

func MakeArray(shape ...int) *Array {
	size := 1
	for i := 0; i < len(shape); i++ {
		if shape[i] <= 0 {
			panic(fmt.Sprintf("invalid shape value: %v", shape))
		}
		size *= shape[i]
	}

	return &Array{
		V:     make([]float32, size),
		Shape: shape,
	}
}

type ActivationType int

const (
	ReLU ActivationType = iota
	Linear
	Sigmoid
)

type LossFunctionType int

const (
	SparseCategoricalCrossEntropyFromLogits LossFunctionType = iota
	MeanSquaredError
)

type Network struct {
	LossFunction LossFunctionType
	Layers       []*Layer
}

func (net *Network) Apply(x []float32, batchSize int) []float32 {
	// Collect max-sized layer output needed.
	maxOutputSize := len(x)
	for l := 0; l < len(net.Layers); l++ {
		if net.Layers[l].OutputSize > maxOutputSize {
			maxOutputSize = net.Layers[l].OutputSize
		}
	}

	a0 := make([]float32, 0, maxOutputSize*batchSize)
	z := make([]float32, 0, maxOutputSize*batchSize)
	a1 := make([]float32, 0, maxOutputSize*batchSize)

	// Copy the input into a0
	a0 = a0[:len(x)]
	copy(a0, x)

	for l := 0; l < len(net.Layers); l++ {
		// Resize our outputs correctly for this layer.
		z = z[:net.Layers[l].OutputSize*batchSize]
		a1 = a1[:net.Layers[l].OutputSize*batchSize]

		net.Layers[l].Apply(a0, z, a1, batchSize)

		// This layer's output becomes the input for the next layer.
		a0, a1 = a1, a0
	}

	return a0
}

func (net *Network) Loss(x, y []float32, batchSize int) float32 {
	a := net.Apply(x, batchSize)

	loss := float32(0)
	switch net.LossFunction {
	case MeanSquaredError:
		loss = net.Layers[0].MeanSquaredErrorLoss(y, a, batchSize)
	case SparseCategoricalCrossEntropyFromLogits:
		loss = net.Layers[0].SparseCategoricalCrossEntropyLoss(y, a, batchSize)
	default:
		panic("unimplemented loss function type")
	}
	return loss
}

func (net *Network) GradientDescent(x, y []float32, batchSize int, alpha float32, steps int) {
	lay := net.Layers[0]

	z := make([]float32, net.Layers[0].OutputSize*batchSize)
	a := make([]float32, net.Layers[0].OutputSize*batchSize)
	dJda := make([]float32, net.Layers[0].OutputSize*batchSize)
	dadz := make([]float32, net.Layers[0].OutputSize*batchSize)
	dJdw := make([]float32, lay.OutputSize*lay.InputSize)
	dJdb := make([]float32, lay.OutputSize*1)
	dJdx := make([]float32, lay.InputSize*batchSize)

	for s := 0; s < steps; s++ {
		// if s < 10 {
		// 	log.Printf("toolkit step=%v w=%v b=%v gradM=%v gradB=%v loss=%v", s, lay.W, lay.B, dJdw, dJdb, net.Loss(x, y, batchSize))
		// }

		net.Layers[0].Apply(x, z, a, batchSize)
		switch net.LossFunction {
		case MeanSquaredError:
			net.Layers[0].MeanSquaredErrorLossGradient(y, a, dJda, batchSize)
		case SparseCategoricalCrossEntropyFromLogits:
			net.Layers[0].SparseCategoricalCrossEntropyLossGradient(y, a, dJda, batchSize)
		default:
			panic("unimplemented loss function type")
		}

		net.Layers[0].Backprop(x, z, dJda, dadz, dJdw, dJdb, dJdx, batchSize)

		for i := 0; i < lay.OutputSize; i++ {
			for j := 0; j < lay.InputSize; j++ {
				lay.W[i*lay.InputSize+j] -= alpha * dJdw[i*lay.InputSize+j]
			}
			lay.B[i] -= alpha * dJdb[i]
		}

		// if s%100000 == 0 {
		// 	log.Printf("toolkit step=%v w=%v b=%v gradM=%v gradB=%v loss=%v", s, lay.W, lay.B, dJdw, dJdb, net.Loss(x, y, batchSize))
		// }
	}
}

type Layer struct {
	Activation ActivationType

	W []float32 // (OutputSize, InputSize)
	B []float32 // (OutputSize, 1)

	InputSize  int
	OutputSize int
}

func MakeDense(activation ActivationType, inputSize, outputSize int) *Layer {
	return &Layer{
		Activation: activation,
		InputSize:  inputSize,
		OutputSize: outputSize,
		W:          make([]float32, outputSize*inputSize),
		B:          make([]float32, outputSize),
	}
}

// Apply the layer in the forward direction.
//
// x (input) is the layer input.  Shape (batchSize, lay.InputSize)
// z (output) is the layer's forward linear output (pre-activation).  Shape (batchSize, lay.OutputSize)
// a (output) is the layer's forward output.  Shape (batchSize, lay.OutputSize)
func (lay *Layer) Apply(x, z, a []float32, batchSize int) {
	if len(x) != lay.InputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(lay.W) != lay.OutputSize*lay.InputSize {
		panic("dimension mismatch")
	}
	if len(lay.B) != lay.OutputSize {
		panic("dimension mismatch")
	}
	if len(z) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(a) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}

	// const eltSize = 4
	// px := unsafe.Pointer(unsafe.SliceData(x))
	// pw := unsafe.Pointer(unsafe.SliceData(lay.W))
	// pb := unsafe.Pointer(unsafe.SliceData(lay.B))
	// pz := unsafe.Pointer(unsafe.SliceData(z))

	// Linear part: z_ik = sum_j w_ij*x_jk + b_i
	//
	// TODO: Make cache-oblivious?  Need benchmarks.
	for k := 0; k < batchSize; k++ {
		for i := 0; i < lay.OutputSize; i++ {
			var sum float32
			for j := 0; j < lay.InputSize; j++ {
				sum += lay.W[i*lay.InputSize+j] * x[k*lay.InputSize+j]
			}

			z[k*lay.OutputSize+i] = sum + lay.B[i]
		}
	}

	switch lay.Activation {
	case ReLU:
		for i := 0; i < len(a); i++ {
			if a[i] < 0 {
				a[i] = 0
			} else {
				a[i] = z[i]
			}
		}
	case Linear:
		for i := 0; i < len(a); i++ {
			a[i] = z[i]
		}
	case Sigmoid:
		for i := 0; i < len(a); i++ {
			a[i] = 1 / (1 + math32.Exp(-z[i]))
		}
	default:
		panic("unhandled activation function")
	}
}

// y is the ground truth output.  Shape (batchSize, lay.OutputSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
func (lay *Layer) MeanSquaredErrorLoss(y, a []float32, batchSize int) float32 {
	loss := float32(0)

	for k := 0; k < batchSize; k++ {
		for i := 0; i < lay.OutputSize; i++ {
			diff := a[k*lay.OutputSize+i] - y[k*lay.OutputSize+i]
			loss += diff * diff / 2 / float32(batchSize) / float32(lay.OutputSize)
		}
	}

	return loss
}

// y is the ground truth output.  Shape (batchSize, lay.OutputSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (scratch) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func (lay *Layer) MeanSquaredErrorLossGradient(y, a, dJda []float32, batchSize int) {
	if len(y) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(a) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(dJda) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	// Hints for bounds check elimination.  Empirically, we seem to need both
	// the panic checks and the access checks.  I believe the panic checks teach
	// the SSA which slices have the same length, and the accesses teach the SSA
	// that the slices are long enough for the accesses.
	_ = y[lay.OutputSize*batchSize-1]
	_ = a[lay.OutputSize*batchSize-1]
	_ = dJda[lay.OutputSize*batchSize-1]

	for k := 0; k < batchSize; k++ {
		for i := 0; i < lay.OutputSize; i++ {
			dJda[k*lay.OutputSize+i] = (a[k*lay.OutputSize+i] - y[k*lay.OutputSize+i]) / float32(batchSize) / float32(lay.OutputSize)
		}
	}
}

// y is the ground truth output.  Shape (batchSize, 1)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
func (lay *Layer) SparseCategoricalCrossEntropyLoss(y, a []float32, batchSize int) float32 {
	loss := float32(0)
	for s := 0; s < batchSize; s++ {
		// Inlined logSumExp over l
		maxa := math32.Inf(-1)
		for l := 0; l < lay.OutputSize; l++ {
			if a[s*lay.OutputSize+l] > maxa {
				maxa = a[s*lay.OutputSize+l]
			}
		}
		suma := maxa
		for l := 0; l < lay.OutputSize; l++ {
			suma += math32.Exp(a[s*lay.OutputSize+l] - maxa)
		}

		for t := 0; t < lay.OutputSize; t++ {
			if y[s] == float32(t) {
				softmax := math32.Exp(a[s*lay.OutputSize+t]-maxa) / suma
				loss += -math32.Log(softmax) / float32(batchSize)
			}
		}
	}

	return loss
}

// y is the ground truth output.  Shape (batchSize, 1)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (scratch) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func (lay *Layer) SparseCategoricalCrossEntropyLossGradient(y, a, dJda []float32, batchSize int) {
	if len(y) != 1*batchSize {
		panic("dimension mismatch")
	}
	if len(a) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(dJda) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}

	for k := 0; k < batchSize; k++ {
		// We are taking softmax(a[l,k]) for all l.
		//
		// For stability, use the identity softmax(v) = softmax(v - c), and
		// subtract the maximimum element of a from every element as we evaluate
		// the softmax.
		//
		// https://stackoverflow.com/questions/42599498/numerically-stable-softmax

		maxa := math32.Inf(-1)
		for l := 0; l < lay.OutputSize; l++ {
			if a[k*lay.OutputSize+l] > maxa {
				maxa = a[k*lay.OutputSize+l]
			}
		}

		var sum float32
		for l := 0; l < lay.OutputSize; l++ {
			sum += math32.Exp(a[k*lay.OutputSize+l] - maxa)
		}

		for i := 0; i < lay.OutputSize; i++ {
			softmax := math32.Exp(a[k*lay.OutputSize+i]-maxa) / sum
			if y[k] == float32(i) {
				dJda[k*lay.OutputSize+i] = (softmax - 1) / float32(batchSize)
			} else {
				dJda[k*lay.OutputSize+i] = (softmax - 0) / float32(batchSize)
			}
		}
	}
}

// x (input) is the layer input.  Shape (batchSize, lay.InputSize)
// z (input) is the layer's forward linear output (pre-activation).  Shape (batchSize, lay.OutputSize)
// dJda (input) is the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
// dadz (scratch) is the gradient of a_ik wrt z_ik.  Shape (batchSize, lay.OutputSize).
// dJdw (output) is the gradient of the loss wrt lay.W.  Shape (lay.OutputSize, lay.InputSize)
// dJdb (output) is the gradient of the loss wrt lay.B.  Shape (lay.OutputSize, 1)
// dJdx (output) is the gradient of the loss wrt x.  Shape (batchSize, lay.InputSize)
func (lay *Layer) Backprop(x, z, dJda, dadz, dJdw, dJdb, dJdx []float32, batchSize int) {
	if len(lay.W) != lay.OutputSize*lay.InputSize {
		panic("dimension mismatch")
	}
	if len(lay.B) != lay.OutputSize {
		panic("dimension mismatch")
	}
	if len(x) != lay.InputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(z) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(dJda) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(dadz) != lay.OutputSize*batchSize {
		panic("dimension mismatch")
	}
	if len(dJdw) != lay.OutputSize*lay.InputSize {
		panic("dimension mismatch")
	}
	if len(dJdb) != lay.OutputSize {
		panic("dimension mismatch")
	}
	if len(dJdx) != lay.InputSize*batchSize {
		panic("dimension mismatch")
	}

	// Compute gradient of activation with respect to z.  This is dependent on
	// the activation function of the layer.
	//
	// TODO: We don't need a scratch matrix for this --- da_ik/dz_ik can be
	// inlined into the weight, bias, and x gradient calculations below.
	for k := 0; k < batchSize; k++ {
		for i := 0; i < lay.OutputSize; i++ {
			switch lay.Activation {
			case Linear:
				dadz[k*lay.OutputSize+i] = 1
			case ReLU:
				if z[k*lay.OutputSize+i] <= 0 {
					dadz[k*lay.OutputSize+i] = 0
				} else {
					dadz[k*lay.OutputSize+i] = 1
				}
			case Sigmoid:
				tmp := math32.Exp(-z[k*lay.OutputSize+i])
				dadz[k*lay.OutputSize+i] = -tmp / (1 + tmp) / (1 + tmp)
			}
		}
	}

	// Compute gradient of loss with respect to weights.
	for i := 0; i < lay.OutputSize; i++ {
		for j := 0; j < lay.InputSize; j++ {
			var grad float32
			for k := 0; k < batchSize; k++ {
				grad += dJda[k*lay.OutputSize+i] * dadz[k*lay.OutputSize+i] * x[k*lay.InputSize+j]
			}
			dJdw[i*lay.InputSize+j] = grad
		}
	}

	// Compute gradient of loss with respect to biases.
	for i := 0; i < lay.OutputSize; i++ {
		var grad float32
		for k := 0; k < batchSize; k++ {
			grad += dJda[k*lay.OutputSize+i] * dadz[k*lay.OutputSize+i]
		}
		dJdb[i] = grad
	}

	// Compute gradient of loss with respect to x.
	for j := 0; j < lay.InputSize; j++ {
		for k := 0; k < batchSize; k++ {
			var grad float32
			for i := 0; i < lay.OutputSize; i++ {
				grad += dJda[k*lay.OutputSize+i] * dadz[k*lay.OutputSize+i] * lay.W[i*lay.OutputSize+j]
			}
			dJdx[k*lay.InputSize+j] = grad
		}
	}
}
