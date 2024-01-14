package toolbox

import (
	"log"

	"github.com/chewxy/math32"
)

// Verify bounds check elimination with
//
//   go build -gcflags="-d=ssa/check_bce" ./toolbox/`

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

func (net *Network) Loss(x, y []float32, batchSize int) float32 {
	z := make([]float32, net.Layers[0].OutputSize*batchSize)
	a := make([]float32, net.Layers[0].OutputSize*batchSize)

	net.Layers[0].Apply(x, z, a, batchSize)

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
		if s < 10 {
			log.Printf("toolkit step=%v w=%v b=%v gradM=%v gradB=%v loss=%v", s, lay.W, lay.B, dJdw, dJdb, net.Loss(x, y, batchSize))
		}

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

		if s%100000 == 0 {
			log.Printf("toolkit step=%v w=%v b=%v gradM=%v gradB=%v loss=%v", s, lay.W, lay.B, dJdw, dJdb, net.Loss(x, y, batchSize))
		}
	}
}

type Layer struct {
	Activation ActivationType

	W []float32 // (OutputSize, InputSize)
	B []float32 // (OutputSize, 1)

	InputSize  int
	OutputSize int
}

// Apply the layer in the forward direction.
//
// x (input) is the layer input.  Sized (lay.InputSize, batchSize)
// z (output) is the layer's forward linear output (pre-activation).  Sized (lay.OutputSize, batchSize)
// a (output) is the layer's forward output.  Sized(lay.OutputSize, batchSize)
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
	for i := 0; i < lay.OutputSize; i++ {
		for k := 0; k < batchSize; k++ {
			var sum float32
			for j := 0; j < lay.InputSize; j++ {
				sum += lay.W[i*lay.InputSize+j] * x[j*batchSize+k]
				// wij := *(*float32)(unsafe.Pointer(uintptr(pw) + uintptr(i*lay.InputSize*eltSize+j*eltSize)))
				// xjk := *(*float32)(unsafe.Pointer(uintptr(px) + uintptr(j*batchSize*eltSize+k*eltSize)))
				// sum += wij * xjk
			}

			z[i*batchSize+k] = sum + lay.B[i]
			// pbi := (*float32)(unsafe.Pointer(uintptr(pb) + uintptr(i*eltSize)))
			// pzik := (*float32)(unsafe.Pointer(uintptr(pz) + uintptr(i*batchSize*eltSize+k*eltSize)))
			// *pzik = sum + *pbi
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

func (lay *Layer) MeanSquaredErrorLoss(y, a []float32, batchSize int) float32 {
	loss := float32(0)
	for i := 0; i < lay.OutputSize; i++ {
		for k := 0; k < batchSize; k++ {
			diff := a[i*batchSize+k] - y[i*batchSize+k]
			loss += diff * diff / 2 / float32(batchSize) / float32(lay.OutputSize)
		}
	}
	return loss
}

// y is the overall model output.  Sized (lay.OutputSize, batchSize)
// a is the layer's forward output.  Sized (lay.OutputSize, batchSize)
// dJda (scratch) is storage for the gradient of the loss wrt a.  Sized (lay.OutputSize, batchSize)
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

	// const eltSize = 4
	// py := unsafe.Pointer(unsafe.SliceData(y))
	// pa := unsafe.Pointer(unsafe.SliceData(a))
	// pdJda := unsafe.Pointer(unsafe.SliceData(dJda))

	// TODO: I cannot for the life of me get these bounds checks to go away.  I
	// have tried strength-reducing the multiplications within the loop.
	//
	// The only thing that works is removing the `-1`s from the access checks
	// above.  But that's wrong --- it will panic when it's actually run because
	// it's accessing beyond the end of the array.
	//
	// I wonder if this is a compiler bug?  It seems to not believe that
	// i*batchSize+k is not always less than lay.OutputSize * batchSize, even
	// after strength-reducing it.
	//
	// For now, I guess rewrite in unsafe.

	// TODO: Unsafe didn't noticeably help the speed --- I guess I need to put
	// some benchmarks in.

	for i := 0; i < lay.OutputSize; i++ {
		for k := 0; k < batchSize; k++ {
			dJda[i*batchSize+k] = (a[i*batchSize+k] - y[i*batchSize+k]) / float32(batchSize) / float32(lay.OutputSize)
			// aik := *(*float32)(unsafe.Pointer(uintptr(pa) + uintptr(i*batchSize*eltSize+k*eltSize)))
			// yik := *(*float32)(unsafe.Pointer(uintptr(py) + uintptr(i*batchSize*eltSize+k*eltSize)))
			// djdaik := (aik - yik) / float32(batchSize) / float32(lay.OutputSize)
			// *(*float32)(unsafe.Pointer(uintptr(pdJda) + uintptr(i*batchSize*eltSize+k*eltSize))) = djdaik
		}
	}
}

func (lay *Layer) SparseCategoricalCrossEntropyLoss(y, a []float32, batchSize int) float32 {
	loss := float32(0)
	for s := 0; s < batchSize; s++ {
		// Inlined logSumExp over l
		maxa := math32.Inf(-1)
		for l := 0; l < lay.OutputSize; l++ {
			if a[l*batchSize+s] > maxa {
				maxa = a[l*batchSize+s]
			}
		}
		suma := maxa
		for l := 0; l < lay.OutputSize; l++ {
			suma += math32.Exp(a[l*batchSize+s] - maxa)
		}

		for t := 0; t < lay.OutputSize; t++ {
			if y[s] == float32(t) {
				softmax := math32.Exp(a[t*batchSize+s]-maxa) / suma
				loss += -math32.Log(softmax) / float32(batchSize)
			}
		}
	}

	return loss
}

// y is the overall model output.  Sized (1, batchSize)
// a is the layer's forward output.  Sized (lay.OutputSize, batchSize)
// dJda (scratch) is storage for the gradient of the loss wrt a.  Sized (lay.OutputSize, batchSize)
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
			if a[l*batchSize+k] > maxa {
				maxa = a[l*batchSize+k]
			}
		}

		var sum float32
		for l := 0; l < lay.OutputSize; l++ {
			sum += math32.Exp(a[l*batchSize+k] - maxa)
		}

		for i := 0; i < lay.OutputSize; i++ {
			softmax := math32.Exp(a[i*batchSize+k]-maxa) / sum
			if y[0*batchSize+k] == float32(i) {
				dJda[i*batchSize+k] = (softmax - 1) / float32(batchSize)
			} else {
				dJda[i*batchSize+k] = (softmax - 0) / float32(batchSize)
			}
		}
	}
}

// x (input) is the layer input.  Sized (lay.InputSize, batchSize)
// z (input) is the layer's forward linear output (pre-activation).  Sized (lay.OutputSize, batchSize)
// dJda (input) is the gradient of the loss wrt a.  Sized (lay.OutputSize, batchSize)
// dadz (scratch) is the gradient of a_ik wrt z_ik.  Sized (lay.OutputSize, batchSize).
// dJdw (output) is the gradient of the loss wrt lay.W.  Sized (lay.OutputSize, lay.InputSize)
// dJdb (output) is the gradient of the loss wrt lay.B.  Sized (lay.OutputSize, 1)
// dJdx (output) is the gradient of the loss wrt x.  Sized (lay.InputSize, batchSize)
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

	// const eltSize = 4
	// pw := unsafe.Pointer(unsafe.SliceData(lay.W))
	// px := unsafe.Pointer(unsafe.SliceData(x))
	// pz := unsafe.Pointer(unsafe.SliceData(z))
	// pdJda := unsafe.Pointer(unsafe.SliceData(dJda))
	// pdadz := unsafe.Pointer(unsafe.SliceData(dadz))
	// pdJdw := unsafe.Pointer(unsafe.SliceData(dJdw))
	// pdJdb := unsafe.Pointer(unsafe.SliceData(dJdb))
	// pdJdx := unsafe.Pointer(unsafe.SliceData(dJdx))

	// Compute gradient of activation with respect to z.  This is dependent on
	// the activation function of the layer.
	//
	// TODO: We don't need a scratch matrix for this --- da_ik/dz_ik can be
	// inlined into the weight, bias, and x gradient calculations below.
	for i := 0; i < lay.OutputSize; i++ {
		for k := 0; k < batchSize; k++ {
			switch lay.Activation {
			case Linear:
				dadz[i*batchSize+k] = 1
				// pdadzik := (*float32)(unsafe.Pointer(uintptr(pdadz) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// *pdadzik = 1
			case ReLU:
				if z[i*batchSize+k] <= 0 {
					dadz[i*batchSize+k] = 0
				} else {
					dadz[i*batchSize+k] = 1
				}
				// pzik := (*float32)(unsafe.Pointer(uintptr(pz) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// pdadzik := (*float32)(unsafe.Pointer(uintptr(pdadz) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// if *pzik <= 0 {
				// 	*pdadzik = 0
				// } else {
				// 	*pdadzik = 1
				// }
			case Sigmoid:
				tmp := math32.Exp(-z[i*batchSize+k])
				dadz[i*batchSize+k] = -tmp / (1 + tmp) / (1 + tmp)
				// pzik := (*float32)(unsafe.Pointer(uintptr(pz) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// tmp := math32.Exp(-(*pzik))
				// pdadzik := (*float32)(unsafe.Pointer(uintptr(pdadz) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// *pdadzik = -tmp / (1 + tmp) / (1 + tmp)
			}
		}
	}

	// Compute gradient of loss with respect to weights.
	for i := 0; i < lay.OutputSize; i++ {
		for j := 0; j < lay.InputSize; j++ {
			var grad float32
			for k := 0; k < batchSize; k++ {
				grad += dJda[i*batchSize+k] * dadz[i*batchSize+k] * x[j*batchSize+k]
				// pdJdaik := (*float32)(unsafe.Pointer(uintptr(pdJda) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// pdadzik := (*float32)(unsafe.Pointer(uintptr(pdadz) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// pxjk := (*float32)(unsafe.Pointer(uintptr(px) + uintptr(j*batchSize*eltSize+k*eltSize)))
				// grad += (*pdJdaik) * (*pdadzik) * (*pxjk)
			}
			dJdw[i*lay.InputSize+j] = grad
			// pdJdwij := (*float32)(unsafe.Pointer(uintptr(pdJdw) + uintptr(i*lay.InputSize*eltSize+j*eltSize)))
			// *pdJdwij = grad
		}
	}

	// Compute gradient of loss with respect to biases.
	for i := 0; i < lay.OutputSize; i++ {
		var grad float32
		for k := 0; k < batchSize; k++ {
			grad += dJda[i*batchSize+k] * dadz[i*batchSize+k]
			// pdJdaik := (*float32)(unsafe.Pointer(uintptr(pdJda) + uintptr(i*batchSize*eltSize+k*eltSize)))
			// pdadzik := (*float32)(unsafe.Pointer(uintptr(pdadz) + uintptr(i*batchSize*eltSize+k*eltSize)))
			// grad += (*pdJdaik) * (*pdadzik)
		}
		dJdb[i] = grad
		// pdJdbij := (*float32)(unsafe.Pointer(uintptr(pdJdb) + uintptr(i*eltSize)))
		// *pdJdbij = grad
	}

	// Compute gradient of loss with respect to x.
	for j := 0; j < lay.InputSize; j++ {
		for k := 0; k < batchSize; k++ {
			var grad float32
			for i := 0; i < lay.OutputSize; i++ {
				grad += dJda[i*batchSize+k] * dadz[i*batchSize+k] * lay.W[i*lay.OutputSize+j]
				// pdJdaik := (*float32)(unsafe.Pointer(uintptr(pdJda) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// pdadzik := (*float32)(unsafe.Pointer(uintptr(pdadz) + uintptr(i*batchSize*eltSize+k*eltSize)))
				// pwij := (*float32)(unsafe.Pointer(uintptr(pw) + uintptr(i*lay.OutputSize*eltSize+j*eltSize)))
				// grad += (*pdJdaik) * (*pdadzik) * (*pwij)
			}
			dJdx[j*batchSize+k] = grad
			// pdJdxjk := (*float32)(unsafe.Pointer(uintptr(pdJdx) + uintptr(j*batchSize*eltSize+k*eltSize)))
			// *pdJdxjk = grad
		}
	}
}
