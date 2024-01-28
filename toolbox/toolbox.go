package toolbox

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"slices"
	"time"
	"unsafe"

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

func MakeScalarAF32(scalar float32) *AF32 {
	return &AF32{
		V:      []float32{scalar},
		Shape0: 1,
		Shape1: 1,
	}
}

func MakeAF32Transpose(in *AF32) *AF32 {
	return &AF32{
		V:      make([]float32, in.Shape0*in.Shape1),
		Shape0: in.Shape1,
		Shape1: in.Shape0,
	}
}

func AF32Transpose(in *AF32, out *AF32) {
	for i := 0; i < in.Shape0; i++ {
		for j := 0; j < in.Shape1; j++ {
			out.Set(j, i, in.At(i, j))
		}
	}
}

func (a *AF32) At1(idx int) float32 {
	pBase := unsafe.Pointer(unsafe.SliceData(a.V))
	pElt := (*float32)(unsafe.Pointer(uintptr(pBase) + uintptr(idx*4)))
	return *pElt
	// return a.V[idx]
}

func (a *AF32) At(idx0, idx1 int) float32 {
	pBase := unsafe.Pointer(unsafe.SliceData(a.V))
	pElt := (*float32)(unsafe.Pointer(uintptr(pBase) + uintptr(idx0*a.Shape1*4+idx1*4)))
	return *pElt
	// return a.V[idx0*a.Shape1+idx1]
}

func (a *AF32) CheckedAt(idx0, idx1 int) float32 {
	return a.V[idx0*a.Shape1+idx1]
}

func (a *AF32) Set1(idx int, v float32) {
	pBase := unsafe.Pointer(unsafe.SliceData(a.V))
	pElt := (*float32)(unsafe.Pointer(uintptr(pBase) + uintptr(idx*4)))
	*pElt = v
	// a.V[idx] = v
}

func (a *AF32) Set(idx0, idx1 int, v float32) {
	pBase := unsafe.Pointer(unsafe.SliceData(a.V))
	pElt := (*float32)(unsafe.Pointer(uintptr(pBase) + uintptr(idx0*a.Shape1*4+idx1*4)))
	*pElt = v
	// a.V[idx0*a.Shape1+idx1] = v
}

func WriteSafeTensors(w io.Writer, tensors map[string]*AF32) error {
	header := map[string]SafeTensorInfo{}
	dataOffset := 0

	keys := []string{}
	for k := range tensors {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	for _, k := range keys {
		begin := dataOffset
		dataOffset += len(tensors[k].V) * 4
		end := dataOffset

		header[k] = SafeTensorInfo{
			DType:       "F32",
			Shape:       []int{tensors[k].Shape0, tensors[k].Shape1},
			DataOffsets: []int{begin, end},
		}
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("while marshaling header: %w", err)
	}

	if err := binary.Write(w, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		return fmt.Errorf("while writing header length: %w", err)
	}

	if _, err := w.Write(headerBytes); err != nil {
		return fmt.Errorf("while writing header: %w", err)
	}

	for _, k := range keys {
		if err := binary.Write(w, binary.LittleEndian, tensors[k].V); err != nil {
			return fmt.Errorf("while writing %s values: %w", k, err)
		}
	}

	return nil
}

func ReadSafeTensors(r io.Reader) (map[string]*AF32, error) {
	rat := r.(io.ReaderAt)

	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("while reading header length: %w", err)
	}

	headerBytes := make([]byte, int(headerLen))
	if _, err := r.Read(headerBytes); err != nil {
		return nil, fmt.Errorf("while reading header: %w", err)
	}

	header := map[string]SafeTensorInfo{}
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("while reading header: %w", err)
	}

	tensors := map[string]*AF32{}
	for k, hdr := range header {
		if hdr.DType != "F32" {
			return nil, fmt.Errorf("unsupported dtype %s", hdr.DType)
		}
		if len(hdr.Shape) != 2 {
			return nil, fmt.Errorf("unsupported shape %v", hdr.Shape)
		}
		for _, s := range hdr.Shape {
			if s < 1 {
				return nil, fmt.Errorf("bad shape %v", hdr.Shape)
			}
		}

		size := hdr.Shape[0] * hdr.Shape[1]
		sizeBytes := size * 4
		valBytes := make([]byte, sizeBytes)
		if _, err := rat.ReadAt(valBytes, 8+int64(headerLen)+int64(hdr.DataOffsets[0])); err != nil {
			return nil, fmt.Errorf("while reading bytes for %s: %w", k, err)
		}

		tensor := &AF32{
			V:      castToF32(valBytes),
			Shape0: hdr.Shape[0],
			Shape1: hdr.Shape[1],
		}

		tensors[k] = tensor
	}

	return tensors, nil
}

func castToF32(b []byte) []float32 {
	f := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
	return f
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
	if y.Shape1 != a.Shape1 {
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
	for k := 0; k < batchSize; k++ {
		// Inlined logSumExp over l
		maxa := math32.Inf(-1)
		for l := 0; l < outputSize; l++ {
			if a.At(k, l) > maxa {
				maxa = a.At(k, l)
			}
		}
		suma := maxa
		for l := 0; l < outputSize; l++ {
			suma += math32.Exp(a.At(k, l) - maxa)
		}

		for i := 0; i < outputSize; i++ {
			if y.At(k, 0) == float32(i) {
				softmax := math32.Exp(a.At(k, i)-maxa) / suma

				// Clamp softmax to make sure the loss is finite.
				//
				// https://stackoverflow.com/a/70608107
				if softmax < 1e-7 {
					softmax = 1e-7
				}
				if softmax > 1-1e-7 {
					softmax = 1 - 1e-7
				}

				loss += -math32.Log(softmax) / float32(denom)
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

	// ref https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

	for k := 0; k < batchSize; k++ {
		// We are taking softmax(a[k, l]) for all l.
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

			// Clamp softmax to make sure the loss is finite.
			//
			// https://stackoverflow.com/a/70608107
			if softmax < 1e-7 {
				softmax = 1e-7
			}
			if softmax > 1-1e-7 {
				softmax = 1 - 1e-7
			}

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

type SafeTensorInfo struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

func (net *Network) LoadTensors(tensors map[string]*AF32) error {
	for l := 0; l < len(net.Layers); l++ {
		weightKey := fmt.Sprintf("net.%d.weights", l)
		weightTensor, ok := tensors[weightKey]
		if !ok {
			return fmt.Errorf("no entry for %s", weightKey)
		}
		gotWeightShape := []int{weightTensor.Shape0, weightTensor.Shape1}
		wantWeightShape := []int{net.Layers[l].OutputSize, net.Layers[l].InputSize}
		if !slices.Equal(gotWeightShape, wantWeightShape) {
			return fmt.Errorf("wrong shape; got %v want %v", gotWeightShape, wantWeightShape)
		}
		net.Layers[l].W = weightTensor

		biasKey := fmt.Sprintf("net.%d.biases", l)
		biasTensor, ok := tensors[biasKey]
		if !ok {
			return fmt.Errorf("no entry for %s", biasKey)
		}
		gotBiasShape := []int{biasTensor.Shape0, biasTensor.Shape1}
		wantBiasShape := []int{net.Layers[l].OutputSize, 1}
		if !slices.Equal(gotBiasShape, wantBiasShape) {
			return fmt.Errorf("wrong shape; got %v want %v", gotBiasShape, wantBiasShape)
		}
		net.Layers[l].B = biasTensor
	}

	return nil
}

func (net *Network) DumpTensors(tensors map[string]*AF32) {
	for l := 0; l < len(net.Layers); l++ {
		tensors[fmt.Sprintf("net.%d.weights", l)] = net.Layers[l].W
		tensors[fmt.Sprintf("net.%d.biases", l)] = net.Layers[l].B
	}
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

		net.Layers[l].Apply(a0, a1, nil) // no need to save activation gradients

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

type AdamEvaluationParameters struct {
	step int

	// Adam parameters
	alpha, beta1, beta2, epsilon float32

	// Updated every step
	beta1T, beta2T float32

	batchSize int

	wTranspose []*AF32

	dadz, a, djda                            []*AF32
	dadzTranspose, aTranspose, djdaTranspose []*AF32

	// The current weight gradients per-layer
	djdw, djdb []*AF32

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

	Timings AdamEvaluationTimings
}

type AdamEvaluationTimings struct {
	Overall         time.Duration
	Forward         time.Duration
	Loss            time.Duration
	Backpropagation time.Duration
	MomentVectors   time.Duration
	WeightUpdate    time.Duration
}

func (t *AdamEvaluationTimings) Reset() {
	t.Overall = 0 * time.Second
	t.Forward = 0 * time.Second
	t.Loss = 0 * time.Second
	t.Backpropagation = 0 * time.Second
	t.MomentVectors = 0 * time.Second
	t.WeightUpdate = 0 * time.Second
}

func (aep *AdamEvaluationParameters) DumpTensors(tensors map[string]*AF32) {
	// This is garbage -- save scalars as 1x1 tensors
	tensors["adam.step"] = MakeScalarAF32(float32(aep.step))
	tensors["adam.alpha"] = MakeScalarAF32(aep.alpha)
	tensors["adam.beta1"] = MakeScalarAF32(aep.beta1)
	tensors["adam.beta2"] = MakeScalarAF32(aep.beta2)
	tensors["adam.epsilon"] = MakeScalarAF32(aep.epsilon)
	tensors["adam.beta1T"] = MakeScalarAF32(aep.beta1T)
	tensors["adam.beta2T"] = MakeScalarAF32(aep.beta2T)
	tensors["adam.batchSize"] = MakeScalarAF32(float32(aep.batchSize))

	// We don't save dadz, a, djda, djdw, djdb because they are scratch
	// variables overwritten at each step.

	// We don't save newMW, newVW, newMB, newVB because they are scratch
	// variables overwritten at each step.

	for l := 0; l < len(aep.oldMW); l++ {
		tensors[fmt.Sprintf("adam.%d.oldMW", l)] = aep.oldMW[l]
		tensors[fmt.Sprintf("adam.%d.oldVW", l)] = aep.oldVW[l]
		tensors[fmt.Sprintf("adam.%d.oldMB", l)] = aep.oldMB[l]
		tensors[fmt.Sprintf("adam.%d.oldVB", l)] = aep.oldVB[l]
	}
}

func loadIntFromTensor(tensors map[string]*AF32, key string) (int, error) {
	tensor, ok := tensors[key]
	if !ok {
		return 0, fmt.Errorf("missing tensor %s", key)
	}
	return int(tensor.At(0, 0)), nil
}

func loadFloat32FromTensor(tensors map[string]*AF32, key string) (float32, error) {
	tensor, ok := tensors[key]
	if !ok {
		return 0, fmt.Errorf("missing tensor %s", key)
	}

	return tensor.At(0, 0), nil
}

func (aep *AdamEvaluationParameters) LoadTensors(tensors map[string]*AF32) error {
	var err error
	aep.step, err = loadIntFromTensor(tensors, "adam.step")
	if err != nil {
		return err
	}
	aep.alpha, err = loadFloat32FromTensor(tensors, "adam.alpha")
	if err != nil {
		return err
	}
	aep.beta1, err = loadFloat32FromTensor(tensors, "adam.beta1")
	if err != nil {
		return err
	}
	aep.beta2, err = loadFloat32FromTensor(tensors, "adam.beta2")
	if err != nil {
		return err
	}
	aep.epsilon, err = loadFloat32FromTensor(tensors, "adam.epsilon")
	if err != nil {
		return err
	}
	aep.beta1T, err = loadFloat32FromTensor(tensors, "adam.beta1T")
	if err != nil {
		return err
	}
	aep.beta2T, err = loadFloat32FromTensor(tensors, "adam.beta2T")
	if err != nil {
		return err
	}
	aep.batchSize, err = loadIntFromTensor(tensors, "adam.batchSize")
	if err != nil {
		return err
	}

	for l := 0; l < len(aep.oldMW); l++ {
		var ok bool
		aep.oldMW[l], ok = tensors[fmt.Sprintf("adam.%d.oldMW", l)]
		if !ok {
			return fmt.Errorf("missing tensor adam.%d.oldMW", l)
		}
		aep.oldVW[l], ok = tensors[fmt.Sprintf("adam.%d.oldVW", l)]
		if !ok {
			return fmt.Errorf("missing tensor adam.%d.oldVW", l)
		}
		aep.oldMB[l], ok = tensors[fmt.Sprintf("adam.%d.oldMB", l)]
		if !ok {
			return fmt.Errorf("missing tensor adam.%d.oldMB", l)
		}
		aep.oldVB[l], ok = tensors[fmt.Sprintf("adam.%d.oldVB", l)]
		if !ok {
			return fmt.Errorf("missing tensor adam.%d.oldVB", l)
		}
	}

	return nil
}

func (net *Network) MakeAdamParameters(alpha float32, batchSize int) *AdamEvaluationParameters {
	aep := &AdamEvaluationParameters{
		alpha:   alpha,
		beta1:   0.9,
		beta2:   0.999,
		epsilon: 1e-7,

		beta1T: 0.9,
		beta2T: 0.999,

		batchSize: batchSize,
	}

	aep.wTranspose = make([]*AF32, len(net.Layers))
	aep.dadz = make([]*AF32, len(net.Layers))
	aep.dadzTranspose = make([]*AF32, len(net.Layers))
	aep.a = make([]*AF32, len(net.Layers))
	aep.aTranspose = make([]*AF32, len(net.Layers))
	aep.djda = make([]*AF32, len(net.Layers))
	aep.djdaTranspose = make([]*AF32, len(net.Layers))
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
		aep.wTranspose[l] = MakeAF32Transpose(net.Layers[l].W)
		aep.dadz[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.dadzTranspose[l] = MakeAF32Transpose(aep.dadz[l])
		aep.a[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.aTranspose[l] = MakeAF32Transpose(aep.a[l])
		aep.djda[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.djdaTranspose[l] = MakeAF32Transpose(aep.djda[l])
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

	return aep
}

// x is the input.  Shape (batchSize, layers[0].InputSize)
// y is the ground truth output.  Shape (batchSize, ?(dependent on loss function))
func (net *Network) Adam(x, y *AF32, alpha float32, steps int) {
	batchSize := x.Shape0

	aep := net.MakeAdamParameters(alpha, batchSize)

	for s := 0; s < steps; s++ {
		net.AdamStep(x, y, aep, 4)
	}
}

func (net *Network) AdamStep(x, y *AF32, aep *AdamEvaluationParameters, threads int) {
	start := time.Now()

	// Create transposed copies of djda, dadz, and a/x.  Backprop calculations of djdw
	// and djdb are better with k being the inner dimension.  Backprop
	// calculation of djdx is better with i as the inner dimension.
	xTranspose := MakeAF32Transpose(x)
	AF32Transpose(x, xTranspose)

	forwardStart := time.Now()

	// Forward application, saving the activation gradients at each layer.
	net.Layers[0].Apply(x, aep.a[0], aep.dadz[0])
	AF32Transpose(aep.a[0], aep.aTranspose[0])
	AF32Transpose(aep.dadz[0], aep.dadzTranspose[0])
	for l := 1; l < len(net.Layers); l++ {
		net.Layers[l].Apply(aep.a[l-1], aep.a[l], aep.dadz[l])
		AF32Transpose(aep.a[l], aep.aTranspose[l])
		AF32Transpose(aep.dadz[l], aep.dadzTranspose[l])
	}

	aep.Timings.Forward += time.Since(forwardStart)

	lossStart := time.Now()

	switch net.LossFunction {
	case MeanSquaredError:
		MeanSquaredErrorLossGradient(y, aep.a[len(net.Layers)-1], aep.djda[len(net.Layers)-1])
	case SparseCategoricalCrossEntropyFromLogits:
		SparseCategoricalCrossEntropyLossGradient(y, aep.a[len(net.Layers)-1], aep.djda[len(net.Layers)-1])
	default:
		panic("unimplemented loss function type")
	}

	aep.Timings.Loss += time.Since(lossStart)

	backpropStart := time.Now()

	// Backprop.  djdx of layer l+i is the djda of layer l.
	//
	// Note that we have to pick the correct per-worker slice of djdw and djdb
	for l := len(net.Layers) - 1; l >= 1; l-- {
		AF32Transpose(aep.djda[l], aep.djdaTranspose[l])

		net.Layers[l].BackpropDjdw(aep.aTranspose[l-1], aep.djdaTranspose[l], aep.dadzTranspose[l], aep.djdw[l])
		net.Layers[l].BackpropDjdb(aep.djdaTranspose[l], aep.dadzTranspose[l], aep.djdb[l])
		net.Layers[l].BackpropDjdx(aep.djda[l], aep.dadz[l], aep.wTranspose[l], aep.djda[l-1])
	}
	AF32Transpose(aep.djda[0], aep.djdaTranspose[0])
	net.Layers[0].BackpropDjdw(xTranspose, aep.djdaTranspose[0], aep.dadzTranspose[0], aep.djdw[0])
	net.Layers[0].BackpropDjdb(aep.djdaTranspose[0], aep.dadzTranspose[0], aep.djdb[0])

	aep.Timings.Backpropagation += time.Since(backpropStart)

	momentVectorsStart := time.Now()

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
		}
	}

	for l := 0; l < len(net.Layers); l++ {
		for i := 0; i < net.Layers[l].OutputSize; i++ {
			djdb := aep.djdb[l].At(i, 0)
			oldmb := aep.oldMB[l].At(i, 0)
			oldvb := aep.oldVB[l].At(i, 0)
			aep.newMB[l].Set(i, 0, beta1*oldmb+(1-beta1)*djdb)
			aep.newVB[l].Set(i, 0, beta2*oldvb+(1-beta2)*djdb*djdb)
		}
	}

	aep.Timings.MomentVectors += time.Since(momentVectorsStart)

	weightUpdateStart := time.Now()

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

		AF32Transpose(net.Layers[l].W, aep.wTranspose[l])
	}

	aep.beta1T *= aep.beta1
	aep.beta2T *= aep.beta2

	aep.oldMW, aep.newMW = aep.newMW, aep.oldMW
	aep.oldMB, aep.newMB = aep.newMB, aep.oldMB
	aep.oldVW, aep.newVW = aep.newVW, aep.oldVW
	aep.oldVB, aep.newVB = aep.newVB, aep.oldVB

	aep.Timings.WeightUpdate += time.Since(weightUpdateStart)

	aep.Timings.Overall += time.Since(start)

	aep.step++
}

type Layer struct {
	Activation ActivationType

	W *AF32 // Shape (OutputSize, InputSize)
	B *AF32 // Shape (OutputSize, 1)

	InputSize  int
	OutputSize int
}

func MakeDense(activation ActivationType, inputSize, outputSize int, r *rand.Rand) *Layer {
	l := &Layer{
		Activation: activation,
		InputSize:  inputSize,
		OutputSize: outputSize,
		W:          MakeAF32(outputSize, inputSize),
		B:          MakeAF32(outputSize, 1),
	}

	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			l.W.Set(i, j, float32(r.NormFloat64())*0.1)
		}
		l.B.Set(i, 0, 0.1)
	}

	return l
}

// Apply the layer in the forward direction.
//
// x (input) is the layer input.  Shape (batchSize, lay.InputSize)
// a (output) is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dadz (output, optional) is the derivative of the activated output wrt the linear output.  Shape (batchSize, lay.OutputSize)
// [sliceMin, sliceMax) is the range of samples we should compute over (used for parallelization)
func (lay *Layer) Apply(x, a, dadz *AF32) {
	batchSize := x.Shape0
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	if x.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if x.Shape1 != inputSize {
		panic("dimension mismatch")
	}
	_ = x.CheckedAt(0, inputSize-1)
	_ = x.CheckedAt(batchSize-1, inputSize-1)

	if a.Shape0 != batchSize {
		panic("dimension mismatch")
	}
	if a.Shape1 != outputSize {
		panic("dimension mismatch")
	}
	_ = a.CheckedAt(batchSize-1, outputSize-1)

	if dadz != nil {
		if dadz.Shape0 != batchSize {
			panic("dimension mismatch")
		}
		if dadz.Shape1 != outputSize {
			panic("dimension mismatch")
		}
		_ = dadz.CheckedAt(batchSize-1, outputSize-1)
	}

	if lay.W.Shape0 != outputSize {
		panic("dimension mismatch")
	}
	if lay.W.Shape1 != inputSize {
		panic("dimension mismatch")
	}
	_ = lay.W.CheckedAt(outputSize-1, inputSize-1)

	if lay.B.Shape0 != outputSize {
		panic("dimension mismatch")
	}
	if lay.B.Shape1 != 1 {
		panic("dimension mismatch")
	}
	_ = lay.B.CheckedAt(outputSize-1, 0)

	iBase := 0
	for i := 0; i < outputSize; i++ {
		kBase := 0
		for k := 0; k < batchSize; k++ {
			z := denseDot2(inputSize, lay.W.V, iBase, x.V, kBase)
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

			kBase += inputSize * 4
		}

		iBase += inputSize * 4
	}
}

// xT (input) is the layer input.  Shape (lay.InputSize, batchSize)
// djdaT (input) is the gradient of the loss wrt a.  Shape (lay.OutputSize, batchSize)
// dadzT (input) is the gradient of a_ik wrt z_ik.  Shape (lay.OutputSize, batchSize)
// dJdw (output) is the gradient of the loss wrt lay.W.  Shape (lay.OutputSize, lay.InputSize)
func (lay *Layer) BackpropDjdw(xT, djdaT, dadzT, djdw *AF32) {
	batchSize := xT.Shape1
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	// This function is equivalent to:
	//
	// for i := 0; i < outputSize; i++ {
	// 	for j := 0; j < inputSize; j++ {
	// 		var grad float32
	// 		for k := 0; k < batchSize; k++ {
	// 			grad += djdaT.At(i, k) * dadzT.At(i, k) * xT.At(j, k)
	// 		}
	// 		djdw.Set(i, j, grad)
	// 	}
	// }

	denseBackpropDjdwSliceKernel(
		batchSize,
		inputSize,
		djdaT.V,
		dadzT.V,
		xT.V,
		djdw.V,
		0, 0,
		outputSize*inputSize,
	)
}

// djdaT (input) is the gradient of the loss wrt a.  Shape (lay.OutputSize, batchSize)
// dadzT (input) is the gradient of a_ik wrt z_ik.  Shape (lay.OutputSize, batchSize)
// dJdb (output) is the gradient of the loss wrt lay.B.  Shape (lay.OutputSize, 1)
func (lay *Layer) BackpropDjdb(djdaT, dadzT, dJdb *AF32) {
	batchSize := djdaT.Shape1
	outputSize := lay.OutputSize

	// Compute gradient of loss with respect to biases.
	iBase := 0
	for i := 0; i < outputSize; i++ {
		grad := denseDot2(batchSize, djdaT.V, iBase, dadzT.V, iBase)
		dJdb.Set(i, 0, grad)

		iBase += batchSize * 4
	}
}

// dJda (input) is the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
// dadz (input) is the gradient of a_ik wrt z_ik.  Shape (batchSize, lay.OutputSize)
// wT (input) is the layer weights, tranposed.  Shape (inputSize, outputSize)
// dJdx (output) is the gradient of the loss wrt x.  Shape (batchSize, lay.InputSize)
func (lay *Layer) BackpropDjdx(djda, dadz, wT, dJdx *AF32) {
	batchSize := djda.Shape0
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	// Compute gradient of loss with respect to x.
	jBase := 0
	for j := 0; j < inputSize; j++ {
		kBase := 0
		for k := 0; k < batchSize; k++ {
			grad := denseDot3(outputSize, djda.V, kBase, dadz.V, kBase, wT.V, jBase)
			dJdx.Set(k, j, grad)

			kBase += outputSize * 4
		}
		jBase += outputSize * 4
	}
}

//go:generate go run asm_dense_dot_2.go -out dense_dot_2.s -stubs stub_dense_dot_2.go
//go:generate go run asm_dense_dot_3.go -out dense_dot_3.s -stubs stub_dense_dot_3.go
//go:generate go run ./asm-generators/asm_dense_backprop_djdw_slice.go -out dense_backprop_djdw_slice.s -stubs stub_dense_backprop_djdw_slice.go
