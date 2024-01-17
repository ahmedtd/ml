package toolbox

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/chewxy/math32"
	"golang.org/x/sync/semaphore"
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
// denom is the total number of samples we will calculate the loss over.  Useful for computing the loss over a set of batches.
func MeanSquaredErrorLoss(y, a *AF32, denom int) float32 {
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
			loss += diff * diff / 2 / float32(denom) / float32(outputSize)
		}
	}

	return loss
}

// y is the ground truth output.  Shape (batchSize, lay.OutputSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (output) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func MeanSquaredErrorLossGradient(y, a, dJda *AF32, sliceMin, sliceMax int) {
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

	for k := sliceMin; k < sliceMax; k++ {
		for i := 0; i < outputSize; i++ {
			grad := (a.At(k, i) - y.At(k, i)) / float32(batchSize) / float32(outputSize)
			dJda.Set(k, i, grad)
		}
	}
}

// y is the ground truth output.  Shape (batchSize, 1)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
func SparseCategoricalCrossEntropyLoss(y, a *AF32, denom int) float32 {
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
				loss += -math32.Log(softmax) / float32(denom)
			}
		}
	}

	return loss
}

// y is the ground truth output.  Shape (batchSize, 1)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (scratch) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func SparseCategoricalCrossEntropyLossGradient(y, a, dJda *AF32, sliceMin, sliceMax int) {
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

	for k := sliceMin; k < sliceMax; k++ {
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

		net.Layers[l].Apply(a0, a1, nil, 0, batchSize) // no need to save activation gradients

		// This layer's output becomes the input for the next layer.
		a0, a1 = a1, a0
	}

	return a0
}

// xs is the input batches. Shape (batchSize, layers[0].InputSize)
// ys is the ground truth output batches.  Shape (batchSize, ?(dependent on loss function))
func (net *Network) Loss(xs, ys []*AF32) float32 {
	numSamples := 0
	for i := 0; i < len(xs); i++ {
		if xs[i].Shape0 != ys[i].Shape0 {
			panic("dimension mismatch")
		}

		numSamples += xs[i].Shape0
	}

	loss := float32(0)
	for i := 0; i < len(xs); i++ {
		a := net.Apply(xs[i])

		switch net.LossFunction {
		case MeanSquaredError:
			loss += MeanSquaredErrorLoss(ys[i], a, numSamples)
		case SparseCategoricalCrossEntropyFromLogits:
			loss += SparseCategoricalCrossEntropyLoss(ys[i], a, numSamples)
		default:
			panic("unimplemented loss function type")
		}
	}

	return loss
}

type Batch struct {
	X *AF32
	Y *AF32
}

// x is the input.  Shape (batchSize, layers[0].InputSize)
// y is the ground truth output.  Shape (batchSize, ?(dependent on loss function))
func (net *Network) GradientDescent(x, y *AF32, alpha float32, steps int) {
	batchSize := x.Shape0

	// Create per-layer evaluation variables.
	//
	// TODO: We can be smarter about most of these, like we are in Apply().

	// TODO: Very interesting.  Even with only one thread (no goroutine
	// launches, splitting the work up into batches dramatically improves
	// performance.)  Obvious in retrospect, I guess.
	//
	// Work Items | Seconds per step
	//          1 | 20
	//          5 | 8.6
	//         10 | 7
	//         20 | 6
	//         30 | 5.2
	//         40 | 5.2
	//
	// This is on the MNIST problem.  Each input sample is 784 floats, and there
	// are 10000 training examples.  When we divide the input into 40 work
	// units, each one has 250 samples, for a work unit size of 196000 floats, or ~765KiB

	numWorkUnits := 300

	dadz := make([]*AF32, len(net.Layers))
	a := make([]*AF32, len(net.Layers))
	djda := make([]*AF32, len(net.Layers))
	for l := 0; l < len(net.Layers); l++ {
		dadz[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		a[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		djda[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
	}

	// djdw and djdb need to have one replica per worker, which we then sum
	// during the gathering step.
	djdw := make([][]*AF32, numWorkUnits)
	djdb := make([][]*AF32, numWorkUnits)
	for w := 0; w < numWorkUnits; w++ {
		djdw[w] = make([]*AF32, len(net.Layers))
		djdb[w] = make([]*AF32, len(net.Layers))
		for l := 0; l < len(net.Layers); l++ {
			djdw[w][l] = MakeAF32(net.Layers[l].OutputSize, net.Layers[l].InputSize)
			djdb[w][l] = MakeAF32(net.Layers[l].OutputSize, 1)
		}
	}

	for s := 0; s < steps; s++ {
		net.GradientDescentStep(x, y, alpha, numWorkUnits, dadz, a, djda, djdw, djdb)
	}
}

func (net *Network) GradientDescentStep(x, y *AF32, alpha float32, numWorkUnits int, dadz, a, djda []*AF32, djdw, djdb [][]*AF32) {
	batchSize := x.Shape0

	// start := time.Now()

	// Launch one goroutine per work unit, but make sure that we only have this
	// many actively running at once.
	sem := semaphore.NewWeighted(4)

	wg := sync.WaitGroup{}
	for w := 0; w < numWorkUnits; w++ {
		w := w

		// Which samples is this worker responsible for?
		sliceSize := batchSize / numWorkUnits
		sliceMin := w * sliceSize
		sliceMax := (w + 1) * sliceSize
		if w == numWorkUnits-1 {
			sliceMax = batchSize
		}

		sem.Acquire(context.TODO(), 1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer sem.Release(1)

			// Forward application, saving the activation gradients at each layer.
			net.Layers[0].Apply(x, a[0], dadz[0], sliceMin, sliceMax)
			for l := 1; l < len(net.Layers); l++ {
				net.Layers[l].Apply(a[l-1], a[l], dadz[l], sliceMin, sliceMax)
			}

			switch net.LossFunction {
			case MeanSquaredError:
				MeanSquaredErrorLossGradient(y, a[len(net.Layers)-1], djda[len(net.Layers)-1], sliceMin, sliceMax)
			case SparseCategoricalCrossEntropyFromLogits:
				SparseCategoricalCrossEntropyLossGradient(y, a[len(net.Layers)-1], djda[len(net.Layers)-1], sliceMin, sliceMax)
			default:
				panic("unimplemented loss function type")
			}

			// Backprop.  djdx of layer l+i is the djda of layer l.
			//
			// Note that we have to pick the correct per-worker slice of djdw and djdb
			for l := len(net.Layers) - 1; l >= 1; l-- {
				net.Layers[l].Backprop(a[l-1], djda[l], dadz[l], djdw[w][l], djdb[w][l], djda[l-1], sliceMin, sliceMax)
			}
			net.Layers[0].Backprop(x, djda[0], dadz[0], djdw[w][0], djdb[w][0], nil, sliceMin, sliceMax)
		}()
	}
	wg.Wait()

	// backpropFinished := time.Now()

	for l := 0; l < len(net.Layers); l++ {
		for i := 0; i < net.Layers[l].OutputSize; i++ {
			for j := 0; j < net.Layers[l].InputSize; j++ {
				// Sum up the gradients across all worker slices.
				var djdwij float32
				for w := 0; w < numWorkUnits; w++ {
					djdwij += djdw[w][l].At(i, j)
				}

				newW := net.Layers[l].W.At(i, j) - alpha*djdwij
				net.Layers[l].W.Set(i, j, newW)
			}

			// Sum up the gradients across all worker slices.
			var djdbi float32
			for w := 0; w < numWorkUnits; w++ {
				djdbi += djdb[w][l].At(i, 0)
			}

			newB := net.Layers[l].B.At(i, 0) - alpha*djdbi
			net.Layers[l].B.Set(i, 0, newB)
		}
	}

	// weightUpdateFinished := time.Now()

	// overallTime := weightUpdateFinished.Sub(start)
	// backpropTime := backpropFinished.Sub(start)
	// backpropPct := backpropTime.Seconds() / overallTime.Seconds() * 100.0
	// weightUpdateTime := weightUpdateFinished.Sub(backpropFinished)
	// weightUpdatePct := weightUpdateTime.Seconds() / overallTime.Seconds() * 100.0
	// log.Printf("Step overall=%v applyandbackprop=%.1f update=%.1f", overallTime, backpropPct, weightUpdatePct)
}

type AdamEvaluationParameters struct {
	step int

	// Adam parameters
	alpha, beta1, beta2, epsilon float32

	// Updated every step
	beta1T, beta2T float32

	batchSize    int
	numWorkUnits int
	numThreads   int

	dadz, a, djda []*AF32

	// The current weight gradients per-layer
	djdw, djdb []*AF32

	// The current weight gradients per-worker, per-layer.  Will be aggregated
	// into djdw and djdb at the end of the step
	workerDjdw, workerDjdb [][]*AF32

	// The first moment vectors for each layer
	oldMW []*AF32
	oldMB []*AF32
	newMW []*AF32
	newMB []*AF32

	// The second moment vectors for each layer
	oldVW []*AF32
	oldVB []*AF32
	newVW []*AF32
	newVB []*AF32

	overall            time.Duration
	gradientCompute    time.Duration
	gradientReassembly time.Duration
	momentVectors      time.Duration
	weightUpdate       time.Duration
}

func (net *Network) MakeAdamParameters(alpha float32, batchSize, numWorkUnits, numThreads int) *AdamEvaluationParameters {
	aep := &AdamEvaluationParameters{
		alpha:   alpha,
		beta1:   0.9,
		beta2:   0.999,
		epsilon: 1e-7,

		beta1T: 0.9,
		beta2T: 0.999,

		batchSize:    batchSize,
		numWorkUnits: numWorkUnits,
		numThreads:   numThreads,
	}

	aep.dadz = make([]*AF32, len(net.Layers))
	aep.a = make([]*AF32, len(net.Layers))
	aep.djda = make([]*AF32, len(net.Layers))
	aep.djdw = make([]*AF32, len(net.Layers))
	aep.djdb = make([]*AF32, len(net.Layers))
	aep.oldMW = make([]*AF32, len(net.Layers))
	aep.oldVW = make([]*AF32, len(net.Layers))
	aep.oldMB = make([]*AF32, len(net.Layers))
	aep.oldVB = make([]*AF32, len(net.Layers))
	aep.newMW = make([]*AF32, len(net.Layers))
	aep.newVW = make([]*AF32, len(net.Layers))
	aep.newMB = make([]*AF32, len(net.Layers))
	aep.newVB = make([]*AF32, len(net.Layers))
	for l := 0; l < len(net.Layers); l++ {
		aep.dadz[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.a[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.djda[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.djdw[l] = MakeAF32(net.Layers[l].OutputSize, net.Layers[l].InputSize)
		aep.djdb[l] = MakeAF32(net.Layers[l].OutputSize, 1)
		aep.oldMW[l] = MakeAF32(net.Layers[l].OutputSize, net.Layers[l].InputSize)
		aep.oldVW[l] = MakeAF32(net.Layers[l].OutputSize, net.Layers[l].InputSize)
		aep.oldMB[l] = MakeAF32(net.Layers[l].OutputSize, 1)
		aep.oldVB[l] = MakeAF32(net.Layers[l].OutputSize, 1)
		aep.newMW[l] = MakeAF32(net.Layers[l].OutputSize, net.Layers[l].InputSize)
		aep.newVW[l] = MakeAF32(net.Layers[l].OutputSize, net.Layers[l].InputSize)
		aep.newMB[l] = MakeAF32(net.Layers[l].OutputSize, 1)
		aep.newVB[l] = MakeAF32(net.Layers[l].OutputSize, 1)
	}

	// djdw and djdb need to have one replica per worker, which we then sum
	// during the gathering step.
	aep.workerDjdw = make([][]*AF32, aep.numWorkUnits)
	aep.workerDjdb = make([][]*AF32, aep.numWorkUnits)
	for w := 0; w < aep.numWorkUnits; w++ {
		aep.workerDjdw[w] = make([]*AF32, len(net.Layers))
		aep.workerDjdb[w] = make([]*AF32, len(net.Layers))
		for l := 0; l < len(net.Layers); l++ {
			aep.workerDjdw[w][l] = MakeAF32(net.Layers[l].OutputSize, net.Layers[l].InputSize)
			aep.workerDjdb[w][l] = MakeAF32(net.Layers[l].OutputSize, 1)
		}
	}

	return aep
}

// x is the input.  Shape (batchSize, layers[0].InputSize)
// y is the ground truth output.  Shape (batchSize, ?(dependent on loss function))
func (net *Network) Adam(x, y *AF32, alpha float32, numWorkUnits, steps int) {
	batchSize := x.Shape0

	aep := net.MakeAdamParameters(alpha, batchSize, numWorkUnits, 4)

	for s := 0; s < steps; s++ {
		net.AdamStep(x, y, aep)
	}
}

func (net *Network) AdamStep(x, y *AF32, aep *AdamEvaluationParameters) {
	batchSize := x.Shape0

	start := time.Now()

	// Launch one goroutine per work unit, but make sure that we only have this
	// many actively running at once.
	sem := semaphore.NewWeighted(int64(aep.numThreads))

	wg := sync.WaitGroup{}
	for w := 0; w < aep.numWorkUnits; w++ {
		w := w

		// Which samples is this worker responsible for?
		sliceSize := batchSize / aep.numWorkUnits
		sliceMin := w * sliceSize
		sliceMax := (w + 1) * sliceSize
		if w == aep.numWorkUnits-1 {
			sliceMax = batchSize
		}

		sem.Acquire(context.TODO(), 1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer sem.Release(1)

			// Forward application, saving the activation gradients at each layer.
			net.Layers[0].Apply(x, aep.a[0], aep.dadz[0], sliceMin, sliceMax)
			for l := 1; l < len(net.Layers); l++ {
				net.Layers[l].Apply(aep.a[l-1], aep.a[l], aep.dadz[l], sliceMin, sliceMax)
			}

			switch net.LossFunction {
			case MeanSquaredError:
				MeanSquaredErrorLossGradient(y, aep.a[len(net.Layers)-1], aep.djda[len(net.Layers)-1], sliceMin, sliceMax)
			case SparseCategoricalCrossEntropyFromLogits:
				SparseCategoricalCrossEntropyLossGradient(y, aep.a[len(net.Layers)-1], aep.djda[len(net.Layers)-1], sliceMin, sliceMax)
			default:
				panic("unimplemented loss function type")
			}

			// Backprop.  djdx of layer l+i is the djda of layer l.
			//
			// Note that we have to pick the correct per-worker slice of djdw and djdb
			for l := len(net.Layers) - 1; l >= 1; l-- {
				net.Layers[l].Backprop(aep.a[l-1], aep.djda[l], aep.dadz[l], aep.workerDjdw[w][l], aep.workerDjdb[w][l], aep.djda[l-1], sliceMin, sliceMax)
			}
			net.Layers[0].Backprop(x, aep.djda[0], aep.dadz[0], aep.workerDjdw[w][0], aep.workerDjdb[w][0], nil, sliceMin, sliceMax)
		}()
	}
	wg.Wait()

	gradientComputeFinished := time.Now()

	// Reassemble the per-worker weight and bias gradients into a single overall
	// weight and bias gradient.
	for l := 0; l < len(net.Layers); l++ {
		for i := 0; i < net.Layers[l].OutputSize; i++ {
			for j := 0; j < net.Layers[l].InputSize; j++ {
				var sum float32
				for w := 0; w < aep.numWorkUnits; w++ {
					sum += aep.workerDjdw[w][l].At(i, j)
				}
				aep.djdw[l].Set(i, j, sum)
			}

			var sum float32
			for w := 0; w < aep.numWorkUnits; w++ {
				sum += aep.workerDjdb[w][l].At(i, 0)
			}
			aep.djdb[l].Set(i, 0, sum)
		}
	}

	gradientReassemblyFinished := time.Now()

	// Compute new Adam moment vectors
	beta1 := aep.beta1
	beta2 := aep.beta2
	for l := 0; l < len(net.Layers); l++ {
		for i := 0; i < net.Layers[l].OutputSize; i++ {
			for j := 0; j < net.Layers[l].InputSize; j++ {
				djdw := aep.djdw[l].At(i, j)
				oldmw := aep.oldMW[l].At(i, j)
				oldvw := aep.oldVW[l].At(i, j)
				aep.newMW[l].Set(i, j, beta1*oldmw+(1-beta1)*djdw)
				aep.newVW[l].Set(i, j, beta2*oldvw+(1-beta2)*djdw*djdw)
			}

			djdb := aep.djdb[l].At(i, 0)
			oldmb := aep.oldMB[l].At(i, 0)
			oldvb := aep.oldVB[l].At(i, 0)
			aep.newMB[l].Set(i, 0, beta1*oldmb+(1-beta1)*djdb)
			aep.newVB[l].Set(i, 0, beta2*oldvb+(1-beta2)*djdb*djdb)
		}
	}

	momentVectorsFinished := time.Now()

	alpha := aep.alpha
	beta1T := aep.beta1T
	beta2T := aep.beta2T
	alphaT := alpha * math32.Sqrt(1-beta2T) / (1 - beta1T)

	for l := 0; l < len(net.Layers); l++ {
		for i := 0; i < net.Layers[l].OutputSize; i++ {
			for j := 0; j < net.Layers[l].InputSize; j++ {
				newW := net.Layers[l].W.At(i, j) - alphaT*aep.newMW[l].At(i, j)/(math32.Sqrt(aep.newVW[l].At(i, j))+aep.epsilon)
				net.Layers[l].W.Set(i, j, newW)
			}

			newB := net.Layers[l].B.At(i, 0) - alphaT*aep.newMB[l].At(i, 0)/(math32.Sqrt(aep.newVB[l].At(i, 0))+aep.epsilon)
			net.Layers[l].B.Set(i, 0, newB)
		}
	}

	aep.beta1T *= aep.beta1
	aep.beta2T *= aep.beta2

	aep.oldMW, aep.newMW = aep.newMW, aep.oldMW
	aep.oldMB, aep.newMB = aep.newMB, aep.oldMB
	aep.oldVW, aep.newVW = aep.newVW, aep.oldVW
	aep.oldVB, aep.newVB = aep.newVB, aep.oldVB

	weightUpdateFinished := time.Now()

	overallTime := weightUpdateFinished.Sub(start)
	aep.overall += overallTime
	aep.gradientCompute += gradientComputeFinished.Sub(start)
	aep.gradientReassembly += gradientReassemblyFinished.Sub(gradientComputeFinished)
	aep.momentVectors += momentVectorsFinished.Sub(gradientReassemblyFinished)
	aep.weightUpdate += weightUpdateFinished.Sub(momentVectorsFinished)
	// gradientComputePct := gradientComputeFinished.Sub(start).Seconds() / overallTime.Seconds() * 100.0
	// gradientReassemblyPct := gradientReassemblyFinished.Sub(gradientComputeFinished).Seconds() / overallTime.Seconds() * 100.0
	// momentVectorsPct := momentVectorsFinished.Sub(gradientReassemblyFinished).Seconds() / overallTime.Seconds() * 100.0
	// weightUpdatePct := weightUpdateFinished.Sub(momentVectorsFinished).Seconds() / overallTime.Seconds() * 100.0
	// log.Printf("Step %d overall=%v gradientcompute=%.1f gradientreassembly=%.1f momentvectors=%.1f weightupdate=%.1f", aep.step, overallTime, gradientComputePct, gradientReassemblyPct, momentVectorsPct, weightUpdatePct)

	aep.step++
}

type Layer struct {
	Activation ActivationType

	W *AF32 // Shape (OutputSize, InputSize)
	B *AF32 // Shape (OutputSize, 1)

	InputSize  int
	OutputSize int
}

func MakeDense(activation ActivationType, inputSize, outputSize int) *Layer {
	l := &Layer{
		Activation: activation,
		InputSize:  inputSize,
		OutputSize: outputSize,
		W:          MakeAF32(outputSize, inputSize),
		B:          MakeAF32(outputSize, 1),
	}

	r := rand.New(rand.NewSource(12345))

	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			l.W.Set(i, j, r.Float32())
		}
		l.B.Set(i, 0, r.Float32())
	}

	return l
}

// Apply the layer in the forward direction.
//
// x (input) is the layer input.  Shape (batchSize, lay.InputSize)
// a (output) is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dadz (output, optional) is the derivative of the activated output wrt the linear output.  Shape (batchSize, lay.OutputSize)
// [sliceMin, sliceMax) is the range of samples we should compute over (used for parallelization)
func (lay *Layer) Apply(x, a, dadz *AF32, sliceMin, sliceMax int) {
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

	if a.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if a.Shape1 != outputSize {
		panic("dimension mismatch")
	}
	_ = a.At(batchSize-1, outputSize-1)

	if dadz != nil {
		if dadz.Shape0 != batchSize {
			panic("dimension mismatch")
		}
		if dadz.Shape1 != outputSize {
			panic("dimension mismatch")
		}
		_ = dadz.At(batchSize-1, outputSize-1)
	}

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
	for k := sliceMin; k < sliceMax; k++ {
		for i := 0; i < outputSize; i++ {
			var z float32
			for j := 0; j < inputSize; j++ {
				z += lay.W.At(i, j) * x.At(k, j)
			}
			z += lay.B.At(i, 0)

			switch lay.Activation {
			case ReLU:
				if z <= 0 {
					a.Set(k, i, 0)
				} else {
					a.Set(k, i, z)
				}
			case Linear:
				a.Set(k, i, z)
			case Sigmoid:
				a.Set(k, i, 1/(1+math32.Exp(-z)))
			default:
				panic("unhandled activation function")
			}

			if dadz != nil {
				switch lay.Activation {
				case Linear:
					dadz.Set(k, i, 1)
				case ReLU:
					if z <= 0 {
						dadz.Set(k, i, 0)
					} else {
						dadz.Set(k, i, 1)
					}
				case Sigmoid:
					tmp := math32.Exp(-z)
					dadz.Set(k, i, -tmp/(1+tmp)/(1+tmp))
				}
			}
		}
	}
}

// x (input) is the layer input.  Shape (batchSize, lay.InputSize)
// dJda (input) is the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
// dadz (input) is the gradient of a_ik wrt z_ik.  Shape (batchSize, lay.OutputSize).
// dJdw (output) is the gradient of the loss wrt lay.W.  Shape (lay.OutputSize, lay.InputSize)
// dJdb (output) is the gradient of the loss wrt lay.B.  Shape (lay.OutputSize, 1)
// dJdx (output) is the gradient of the loss wrt x.  Shape (batchSize, lay.InputSize)
// [sliceMin, sliceMax) is the range of samples we should compute over (used for parallelization)
func (lay *Layer) Backprop(x, dJda, dadz, dJdw, dJdb, dJdx *AF32, sliceMin, sliceMax int) {
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	// Compute gradient of loss with respect to weights.
	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			var grad float32
			for k := sliceMin; k < sliceMax; k++ {
				grad += dJda.At(k, i) * dadz.At(k, i) * x.At(k, j)
			}
			dJdw.Set(i, j, grad)
		}
	}

	// Compute gradient of loss with respect to biases.
	for i := 0; i < outputSize; i++ {
		var grad float32
		for k := sliceMin; k < sliceMax; k++ {
			grad += dJda.At(k, i) * dadz.At(k, i)
		}
		dJdb.Set(i, 0, grad)
	}

	// Compute gradient of loss with respect to x.  Optional because we don't do
	// it for layer 0.
	if dJdx != nil {
		for j := 0; j < inputSize; j++ {
			for k := sliceMin; k < sliceMax; k++ {
				var grad float32
				for i := 0; i < lay.OutputSize; i++ {
					grad += dJda.At(k, i) * dadz.At(k, i) * lay.W.At(i, j)
				}
				dJdx.Set(k, j, grad)
			}
		}
	}

}
