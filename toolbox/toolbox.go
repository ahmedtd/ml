package toolbox

import (
	"fmt"

	"github.com/chewxy/math32"
)

// Verify bounds check elimination with
//
//   go build -gcflags="-d=ssa/check_bce" ./toolbox/

type AF32 struct {
	V      []float32
	Shape0 int
	Shape1 int
}

func MakeAF32(shape0, shape1 int) *AF32 {
	if shape0 <= 0 {
		panic(fmt.Sprintf("invalid shape value: %v", shape0))
	}
	if shape1 <= 0 {
		panic(fmt.Sprintf("invalid shape value: %v", shape1))
	}

	return &AF32{
		V:      make([]float32, shape0*shape1),
		Shape0: shape0,
		Shape1: shape1,
	}
}

func (a *AF32) At(idx0, idx1 int) float32 {
	return a.V[idx0*a.Shape1+idx1]
}

func (a *AF32) Set(idx0, idx1 int, v float32) {
	a.V[idx0*a.Shape1+idx1] = v
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

// y is the ground truth output.  Shape (batchSize, lay.OutputSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
func MeanSquaredErrorLoss(y, a *AF32) float32 {
	if y.Shape0 != a.Shape0 {
		panic("dimension mismatch")
	}
	if y.Shape1 != y.Shape1 {
		panic("dimension mismatch")
	}

	batchSize := y.Shape0
	outputSize := y.Shape1

	loss := float32(0)

	for k := 0; k < batchSize; k++ {
		for i := 0; i < outputSize; i++ {
			diff := a.At(k, i) - y.At(k, i)
			loss += diff * diff / 2 / float32(batchSize) / float32(outputSize)
		}
	}

	return loss
}

// y is the ground truth output.  Shape (batchSize, lay.OutputSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (output) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func MeanSquaredErrorLossGradient(y, a, dJda *AF32) {
	batchSize := a.Shape0
	outputSize := a.Shape1
	_ = a.At(batchSize-1, outputSize-1)

	if y.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if y.Shape1 != outputSize {
		panic("dimension mismatch")
	}
	_ = y.At(batchSize-1, outputSize-1)

	if dJda.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if dJda.Shape1 != outputSize {
		panic("dimension mismatch")
	}
	_ = dJda.At(batchSize-1, outputSize-1)

	for k := 0; k < batchSize; k++ {
		for i := 0; i < outputSize; i++ {
			grad := (a.At(k, i) - y.At(k, i)) / float32(batchSize) / float32(outputSize)
			dJda.Set(k, i, grad)
		}
	}
}

// y is the ground truth output.  Shape (batchSize, 1)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
func SparseCategoricalCrossEntropyLoss(y, a *AF32) float32 {
	batchSize := a.Shape0
	outputSize := a.Shape1
	_ = a.At(batchSize-1, outputSize-1)

	if y.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if y.Shape1 != 1 {
		panic("dimension mismatch")
	}
	_ = y.At(batchSize-1, 0)

	loss := float32(0)
	for s := 0; s < batchSize; s++ {
		// Inlined logSumExp over l
		maxa := math32.Inf(-1)
		for l := 0; l < outputSize; l++ {
			if a.At(s, l) > maxa {
				maxa = a.At(s, l)
			}
		}
		suma := maxa
		for l := 0; l < outputSize; l++ {
			suma += math32.Exp(a.At(s, l) - maxa)
		}

		for t := 0; t < outputSize; t++ {
			if y.At(s, 0) == float32(t) {
				softmax := math32.Exp(a.At(s, t)-maxa) / suma
				loss += -math32.Log(softmax) / float32(batchSize)
			}
		}
	}

	return loss
}

// y is the ground truth output.  Shape (batchSize, 1)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (scratch) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func SparseCategoricalCrossEntropyLossGradient(y, a, dJda *AF32) {
	batchSize := a.Shape0
	outputSize := a.Shape1
	_ = a.At(batchSize-1, outputSize-1)

	if y.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if y.Shape1 != 1 {
		panic("dimension mismatch")
	}
	_ = y.At(batchSize-1, 0)

	if dJda.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if dJda.Shape1 != outputSize {
		panic("dimension mismatch")
	}
	_ = dJda.At(batchSize-1, outputSize-1)

	for k := 0; k < batchSize; k++ {
		// We are taking softmax(a[l,k]) for all l.
		//
		// For stability, use the identity softmax(v) = softmax(v - c), and
		// subtract the maximimum element of a from every element as we evaluate
		// the softmax.
		//
		// https://stackoverflow.com/questions/42599498/numerically-stable-softmax

		maxa := math32.Inf(-1)
		for l := 0; l < outputSize; l++ {
			if a.At(k, l) > maxa {
				maxa = a.At(k, l)
			}
		}

		var sum float32
		for l := 0; l < outputSize; l++ {
			sum += math32.Exp(a.At(k, l) - maxa)
		}

		for i := 0; i < outputSize; i++ {
			softmax := math32.Exp(a.At(k, i)-maxa) / sum
			if y.At(k, 0) == float32(i) {
				dJda.Set(k, i, (softmax-1)/float32(batchSize))
			} else {
				dJda.Set(k, i, (softmax-0)/float32(batchSize))
			}
		}
	}
}

type Network struct {
	LossFunction LossFunctionType
	Layers       []*Layer
}

// x is the input.  Shape (batchSize, layers[0].InputSize)
func (net *Network) Apply(x *AF32) *AF32 {
	batchSize := x.Shape0

	// Collect max-sized layer output needed.
	maxOutputSize := x.Shape1
	for l := 0; l < len(net.Layers); l++ {
		if net.Layers[l].OutputSize > maxOutputSize {
			maxOutputSize = net.Layers[l].OutputSize
		}
	}

	// Make this in a weird way because we're going to keep resizing them as we
	// move forward through the layers.
	a0 := &AF32{
		V:      make([]float32, 0, batchSize*maxOutputSize),
		Shape0: 0,
		Shape1: 0,
	}
	z := &AF32{
		V:      make([]float32, 0, batchSize*maxOutputSize),
		Shape0: 0,
		Shape1: 0,
	}
	a1 := &AF32{
		V:      make([]float32, 0, batchSize*maxOutputSize),
		Shape0: 0,
		Shape1: 1,
	}

	// Copy the input into a0
	a0.V = a0.V[:batchSize*x.Shape1]
	a0.Shape0 = batchSize
	a0.Shape1 = x.Shape1
	copy(a0.V, x.V)

	for l := 0; l < len(net.Layers); l++ {
		// Resize our outputs correctly for this layer.
		z.V = z.V[:batchSize*net.Layers[l].OutputSize]
		z.Shape0 = batchSize
		z.Shape1 = net.Layers[l].OutputSize
		a1.V = a1.V[:batchSize*net.Layers[l].OutputSize]
		a1.Shape0 = batchSize
		a1.Shape1 = net.Layers[l].OutputSize

		net.Layers[l].Apply(a0, z, a1)

		// This layer's output becomes the input for the next layer.
		a0, a1 = a1, a0
	}

	return a0
}

// x is the input. Shape (batchSize, layers[0].InputSize)
// y is the ground truth output.  Shape (batchSize, ?(dependent on loss function))
func (net *Network) Loss(x, y *AF32) float32 {
	if x.Shape0 != y.Shape0 {
		panic("dimension mismatch")
	}

	a := net.Apply(x)

	loss := float32(0)
	switch net.LossFunction {
	case MeanSquaredError:
		loss = MeanSquaredErrorLoss(y, a)
	case SparseCategoricalCrossEntropyFromLogits:
		loss = SparseCategoricalCrossEntropyLoss(y, a)
	default:
		panic("unimplemented loss function type")
	}
	return loss
}

// x is the input.  Shape (batchSize, layers[0].InputSize)
// y is the ground truth output.  Shape (batchSize, ?(dependent on loss function))
func (net *Network) GradientDescent(x, y *AF32, alpha float32, steps int) {
	batchSize := x.Shape0

	lay := net.Layers[0]

	z := MakeAF32(batchSize, net.Layers[0].OutputSize)
	a := MakeAF32(batchSize, net.Layers[0].OutputSize)
	dJda := MakeAF32(batchSize, net.Layers[0].OutputSize)
	dadz := MakeAF32(batchSize, net.Layers[0].OutputSize)
	dJdw := MakeAF32(net.Layers[0].OutputSize, net.Layers[0].InputSize)
	dJdb := MakeAF32(net.Layers[0].OutputSize, 1)
	dJdx := MakeAF32(batchSize, lay.InputSize)

	for s := 0; s < steps; s++ {
		// if s < 10 {
		// 	log.Printf("toolkit step=%v w=%v b=%v gradM=%v gradB=%v loss=%v", s, lay.W, lay.B, dJdw, dJdb, net.Loss(x, y, batchSize))
		// }

		net.Layers[0].Apply(x, z, a)
		switch net.LossFunction {
		case MeanSquaredError:
			MeanSquaredErrorLossGradient(y, a, dJda)
		case SparseCategoricalCrossEntropyFromLogits:
			SparseCategoricalCrossEntropyLossGradient(y, a, dJda)
		default:
			panic("unimplemented loss function type")
		}

		net.Layers[0].Backprop(x, z, dJda, dadz, dJdw, dJdb, dJdx)

		for i := 0; i < lay.OutputSize; i++ {
			for j := 0; j < lay.InputSize; j++ {
				newW := lay.W.At(i, j) - alpha*dJdw.At(i, j)
				lay.W.Set(i, j, newW)
			}
			newB := lay.B.At(i, 0) - alpha*dJdb.At(i, 0)
			lay.B.Set(i, 0, newB)
		}

		// if s%100000 == 0 {
		// 	log.Printf("toolkit step=%v w=%v b=%v gradM=%v gradB=%v loss=%v", s, lay.W, lay.B, dJdw, dJdb, net.Loss(x, y, batchSize))
		// }
	}
}

type Layer struct {
	Activation ActivationType

	W *AF32 // Shape (OutputSize, InputSize)
	B *AF32 // Shape (OutputSize, 1)

	InputSize  int
	OutputSize int
}

func MakeDense(activation ActivationType, inputSize, outputSize int) *Layer {
	return &Layer{
		Activation: activation,
		InputSize:  inputSize,
		OutputSize: outputSize,
		W:          MakeAF32(outputSize, inputSize),
		B:          MakeAF32(outputSize, 1),
	}
}

// Apply the layer in the forward direction.
//
// x (input) is the layer input.  Shape (batchSize, lay.InputSize)
// z (output) is the layer's forward linear output (pre-activation).  Shape (batchSize, lay.OutputSize)
// a (output) is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dadz (output, optional) is the derivative of the activated output wrt the linear output.  Shape (batchSize, lay.OutputSize)
func (lay *Layer) Apply(x, z, a *AF32) {
	batchSize := x.Shape0
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	if x.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if x.Shape1 != inputSize {
		panic("dimension mismatch")
	}
	_ = x.At(batchSize-1, inputSize-1)

	if z.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if z.Shape1 != outputSize {
		panic("dimension mismatch")
	}
	_ = z.At(batchSize-1, outputSize-1)

	if a.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if a.Shape1 != outputSize {
		panic("dimension mismatch")
	}
	_ = a.At(batchSize-1, outputSize-1)

	if lay.W.Shape0 != outputSize {
		panic("dimension mismatch")
	}
	if lay.W.Shape1 != inputSize {
		panic("dimension mismatch")
	}
	_ = lay.W.At(outputSize-1, inputSize-1)

	if lay.B.Shape0 != outputSize {
		panic("dimension mismatch")
	}
	if lay.B.Shape1 != 1 {
		panic("dimension mismatch")
	}
	_ = lay.B.At(outputSize-1, 0)

	// Linear part: z_ik = sum_j w_ij*x_jk + b_i
	//
	// TODO: Make cache-oblivious?  Need benchmarks.
	for k := 0; k < batchSize; k++ {
		for i := 0; i < outputSize; i++ {
			var sum float32
			for j := 0; j < inputSize; j++ {
				sum += lay.W.At(i, j) * x.At(k, j)
			}
			z.Set(k, i, sum+lay.B.At(i, 0))
		}
	}

	switch lay.Activation {
	case ReLU:
		for k := 0; k < batchSize; k++ {
			for i := 0; i < outputSize; i++ {
				if z.At(k, i) < 0 {
					a.Set(k, i, 0)
				} else {
					a.Set(k, i, z.At(k, i))
				}
			}
		}
	case Linear:
		for k := 0; k < batchSize; k++ {
			for i := 0; i < outputSize; i++ {
				a.Set(k, i, z.At(k, i))
			}
		}
	case Sigmoid:
		for k := 0; k < batchSize; k++ {
			for i := 0; i < outputSize; i++ {
				z.Set(k, i, 1/(1+math32.Exp(-z.At(k, i))))
			}
		}
	default:
		panic("unhandled activation function")
	}
}

// x (input) is the layer input.  Shape (batchSize, lay.InputSize)
// z (input) is the layer's forward linear output (pre-activation).  Shape (batchSize, lay.OutputSize)
// dJda (input) is the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
// dadz (scratch) is the gradient of a_ik wrt z_ik.  Shape (batchSize, lay.OutputSize).
// dJdw (output) is the gradient of the loss wrt lay.W.  Shape (lay.OutputSize, lay.InputSize)
// dJdb (output) is the gradient of the loss wrt lay.B.  Shape (lay.OutputSize, 1)
// dJdx (output) is the gradient of the loss wrt x.  Shape (batchSize, lay.InputSize)
func (lay *Layer) Backprop(x, z, dJda, dadz, dJdw, dJdb, dJdx *AF32) {
	batchSize := x.Shape0
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	// TODO: Consistency checks.

	// Compute gradient of activation with respect to z.  This is dependent on
	// the activation function of the layer.
	//
	// TODO: We don't need a scratch matrix for this --- da_ik/dz_ik can be
	// inlined into the weight, bias, and x gradient calculations below.
	for k := 0; k < batchSize; k++ {
		for i := 0; i < outputSize; i++ {
			switch lay.Activation {
			case Linear:
				dadz.Set(k, i, 1)
			case ReLU:
				if z.At(k, i) <= 0 {
					dadz.Set(k, i, 0)
				} else {
					dadz.Set(k, i, 1)
				}
			case Sigmoid:
				tmp := math32.Exp(-z.At(k, i))
				dadz.Set(k, i, -tmp/(1+tmp)/(1+tmp))
			}
		}
	}

	// Compute gradient of loss with respect to weights.
	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			var grad float32
			for k := 0; k < batchSize; k++ {
				grad += dJda.At(k, i) * dadz.At(k, i) * x.At(k, j)
			}
			dJdw.Set(i, j, grad)
		}
	}

	// Compute gradient of loss with respect to biases.
	for i := 0; i < outputSize; i++ {
		var grad float32
		for k := 0; k < batchSize; k++ {
			grad += dJda.At(k, i) * dadz.At(k, i)
		}
		dJdb.Set(i, 0, grad)
	}

	// Compute gradient of loss with respect to x.
	for j := 0; j < inputSize; j++ {
		for k := 0; k < outputSize; k++ {
			var grad float32
			for i := 0; i < lay.OutputSize; i++ {
				grad += dJda.At(k, i) * dadz.At(k, i) * lay.W.At(i, j)
			}
			dJdx.Set(k, j, grad)
		}
	}
}
