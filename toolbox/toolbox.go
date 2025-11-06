package toolbox

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"simd"
	"slices"
	"time"
	"unsafe"

	"github.com/chewxy/math32"
)

// Verify bounds check elimination with
//
//   go build -gcflags="-d=ssa/check_bce" ./toolbox/

type AF32 struct {
	V     []float32
	Shape []int
}

func MakeAF32(shape ...int) *AF32 {
	for _, s := range shape {
		if s <= 0 {
			panic(fmt.Sprintf("invalid shape: %v", shape))
		}
	}
	size := 1
	for _, s := range shape {
		size *= s
	}

	return &AF32{
		V:     make([]float32, size),
		Shape: shape,
	}
}

func MakeScalarAF32(scalar float32) *AF32 {
	return &AF32{
		V:     []float32{scalar},
		Shape: []int{1},
	}
}

func AF32Copy(in *AF32) *AF32 {
	shapeCopy := make([]int, len(in.Shape))
	copy(shapeCopy, in.Shape)
	return &AF32{
		V:     make([]float32, len(in.V)),
		Shape: shapeCopy,
	}
}

func AF32Transpose(in *AF32, out *AF32) {
	if len(in.Shape) != 2 {
		panic("cannot transpose if len(shape) != 2")
	}
	if len(in.V) != len(out.V) {
		panic("output storage is not correctly sized to store the transpose of the input")
	}
	out.Shape = []int{in.Shape[1], in.Shape[0]}

	for i := 0; i < in.Shape[0]; i++ {
		for j := 0; j < in.Shape[1]; j++ {
			out.Set2(j, i, in.At2(i, j))
		}
	}
}

// AF32Reshape reshapes the input tensor.  The overall number of elements must
// be the same.  The returned tensor shares storage with the input tensor (no
// data is copied).
func AF32Reshape(a *AF32, shape ...int) *AF32 {
	newSize := 1
	for _, s := range shape {
		if s <= 0 {
			panic(fmt.Sprintf("invalid shape: %v", shape))
		}
		newSize *= s
	}

	if newSize != len(a.V) {
		panic("invalid reshape")
	}

	return &AF32{
		V:     a.V,
		Shape: shape,
	}
}

func (a *AF32) At1(idx int) float32 {
	pBase := unsafe.Pointer(unsafe.SliceData(a.V))
	pElt := (*float32)(unsafe.Pointer(uintptr(pBase) + uintptr(idx*4)))
	return *pElt
	// return a.V[idx]
}

func (a *AF32) At2(idx0, idx1 int) float32 {
	if len(a.Shape) != 2 {
		panic("At2() invalid for len(shape) != 2")
	}
	return a.V[idx0*a.Shape[1]+idx1]
}

func (a *AF32) At3(idx0, idx1, idx2 int) float32 {
	if len(a.Shape) != 3 {
		panic("At3() invalid for len(shape) != 3")
	}
	return a.V[idx0*a.Shape[1]*a.Shape[2]+idx1*a.Shape[2]+idx2]
}

func (a *AF32) CheckedAt3(idx0, idx1, idx2 int) float32 {
	return a.V[idx0*a.Shape[1]*a.Shape[2]+idx1*a.Shape[2]+idx2]
}

func (a *AF32) Set1(idx int, v float32) {
	pBase := unsafe.Pointer(unsafe.SliceData(a.V))
	pElt := (*float32)(unsafe.Pointer(uintptr(pBase) + uintptr(idx*4)))
	*pElt = v
	// a.V[idx] = v
}

func (a *AF32) Set2(idx0, idx1 int, v float32) {
	if len(a.Shape) != 2 {
		panic("Set2() invalid for len(shape) != 2")
	}
	a.V[idx0*a.Shape[1]+idx1] = v
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
			Shape:       tensors[k].Shape,
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
		if len(hdr.Shape) > 3 {
			return nil, fmt.Errorf("unsupported shape %v", hdr.Shape)
		}

		size := 1
		for _, s := range hdr.Shape {
			if s < 1 {
				return nil, fmt.Errorf("bad shape %v", hdr.Shape)
			}
			size *= s
		}

		sizeBytes := size * 4
		valBytes := make([]byte, sizeBytes)
		if _, err := rat.ReadAt(valBytes, 8+int64(headerLen)+int64(hdr.DataOffsets[0])); err != nil {
			return nil, fmt.Errorf("while reading bytes for %s: %w", k, err)
		}

		tensor := &AF32{
			V:     castToF32(valBytes),
			Shape: hdr.Shape,
		}
		if len(hdr.Shape) <= 3 {
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
	if len(y.Shape) != 2 {
		panic("len(y.Shape) != 2")
	}
	if len(a.Shape) != 2 {
		panic("len(a.Shape) != 2")
	}
	if !slices.Equal(y.Shape, a.Shape) {
		panic("y and a must have same shape")
	}

	batchSize := y.Shape[0]
	outputSize := y.Shape[1]

	loss := float32(0)

	for k := 0; k < batchSize; k++ {
		for i := 0; i < outputSize; i++ {
			diff := a.At2(k, i) - y.At2(k, i)
			loss += diff * diff / 2 / float32(denom) / float32(outputSize)
		}
	}

	return loss
}

// y is the ground truth output.  Shape (batchSize, lay.OutputSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (output) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func MeanSquaredErrorLossGradient(y, a, dJda *AF32) {
	if len(y.Shape) != 2 {
		panic("len(y.Shape) != 2")
	}
	if !slices.Equal(y.Shape, a.Shape) {
		panic("y and a must have same shape")
	}
	if !slices.Equal(y.Shape, dJda.Shape) {
		panic("y and dJda must have same shape")
	}

	batchSize := a.Shape[0]
	outputSize := a.Shape[1]

	// Hints to help with bounds-check elimination
	_ = a.At2(batchSize-1, outputSize-1)
	_ = y.At2(batchSize-1, outputSize-1)
	_ = dJda.At2(batchSize-1, outputSize-1)

	for k := 0; k < batchSize; k++ {
		for i := 0; i < outputSize; i++ {
			grad := (a.At2(k, i) - y.At2(k, i)) / float32(batchSize) / float32(outputSize)
			dJda.Set2(k, i, grad)
		}
	}
}

// y is the ground truth output.  Shape (batchSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
func SparseCategoricalCrossEntropyLoss(y, a *AF32, denom int) float32 {
	if len(a.Shape) != 2 {
		panic("len(a.Shape) != 2")
	}
	batchSize := a.Shape[0]
	outputSize := a.Shape[1]

	if !slices.Equal(y.Shape, []int{batchSize}) {
		panic("y.Shape != {batchSize}")
	}

	loss := float32(0)
	for k := 0; k < batchSize; k++ {
		// Inlined logSumExp over l
		maxa := math32.Inf(-1)
		for l := 0; l < outputSize; l++ {
			if a.At2(k, l) > maxa {
				maxa = a.At2(k, l)
			}
		}
		suma := maxa
		for l := 0; l < outputSize; l++ {
			suma += math32.Exp(a.At2(k, l) - maxa)
		}

		for i := 0; i < outputSize; i++ {
			if y.At1(k) == float32(i) {
				softmax := math32.Exp(a.At2(k, i)-maxa) / suma

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

// y is the ground truth output.  Shape (batchSize)
// a is the layer's forward output.  Shape (batchSize, lay.OutputSize)
// dJda (scratch) is storage for the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
func SparseCategoricalCrossEntropyLossGradient(y, a, dJda *AF32) {
	if len(a.Shape) != 2 {
		panic("len(a.Shape) != 2")
	}
	batchSize := a.Shape[0]
	outputSize := a.Shape[1]

	if !slices.Equal(y.Shape, []int{batchSize}) {
		panic("y.Shape != {batchSize}")
	}
	if !slices.Equal(dJda.Shape, []int{batchSize, outputSize}) {
		panic("dJda.Shape != {batchSize, outputSize}")
	}

	// Hints for bounds-check elimination.
	_ = a.At2(batchSize-1, outputSize-1)
	_ = y.At1(batchSize - 1)
	_ = dJda.At2(batchSize-1, outputSize-1)

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
			if a.At2(k, l) > maxa {
				maxa = a.At2(k, l)
			}
		}

		var sum float32
		for l := 0; l < outputSize; l++ {
			sum += math32.Exp(a.At2(k, l) - maxa)
		}

		for i := 0; i < outputSize; i++ {
			softmax := math32.Exp(a.At2(k, i)-maxa) / sum

			// Clamp softmax to make sure the loss is finite.
			//
			// https://stackoverflow.com/a/70608107
			if softmax < 1e-7 {
				softmax = 1e-7
			}
			if softmax > 1-1e-7 {
				softmax = 1 - 1e-7
			}

			if y.At1(k) == float32(i) {
				dJda.Set2(k, i, (softmax-1)/float32(batchSize))
			} else {
				dJda.Set2(k, i, (softmax-0)/float32(batchSize))
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
		wantWeightShape := []int{net.Layers[l].OutputSize, net.Layers[l].InputSize}
		if !slices.Equal(weightTensor.Shape, wantWeightShape) {
			return fmt.Errorf("wrong shape; got %v want %v", weightTensor.Shape, wantWeightShape)
		}
		net.Layers[l].W = weightTensor

		biasKey := fmt.Sprintf("net.%d.biases", l)
		biasTensor, ok := tensors[biasKey]
		if !ok {
			return fmt.Errorf("no entry for %s", biasKey)
		}
		wantBiasShape := []int{net.Layers[l].OutputSize, 1}
		if !slices.Equal(biasTensor.Shape, wantBiasShape) {
			return fmt.Errorf("wrong shape; got %v want %v", biasTensor.Shape, wantBiasShape)
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
	batchSize := x.Shape[0]

	// Collect max-sized layer output needed.
	maxOutputSize := x.Shape[1]
	for l := 0; l < len(net.Layers); l++ {
		if net.Layers[l].OutputSize > maxOutputSize {
			maxOutputSize = net.Layers[l].OutputSize
		}
	}

	// Make this in a weird way because we're going to keep resizing them as we
	// move forward through the layers.
	a0 := &AF32{
		V:     make([]float32, 0, batchSize*maxOutputSize),
		Shape: []int{0, 0},
	}
	z := &AF32{
		V:     make([]float32, 0, batchSize*maxOutputSize),
		Shape: []int{0, 0},
	}
	a1 := &AF32{
		V:     make([]float32, 0, batchSize*maxOutputSize),
		Shape: []int{0, 0},
	}

	// Copy the input into a0
	a0.V = a0.V[:batchSize*x.Shape[1]]
	a0.Shape[0] = batchSize
	a0.Shape[1] = x.Shape[1]
	copy(a0.V, x.V)

	for l := 0; l < len(net.Layers); l++ {
		// Resize our outputs correctly for this layer.
		z.V = z.V[:batchSize*net.Layers[l].OutputSize]
		z.Shape[0] = batchSize
		z.Shape[1] = net.Layers[l].OutputSize
		a1.V = a1.V[:batchSize*net.Layers[l].OutputSize]
		a1.Shape[0] = batchSize
		a1.Shape[1] = net.Layers[l].OutputSize

		net.Layers[l].Apply(a0, a1, nil) // no need to save activation gradients

		// This layer's output becomes the input for the next layer.
		a0, a1 = a1, a0
	}

	return a0
}

// xs is the input batches. Shape (batchSize, layers[0].InputSize)
// ys is the ground truth output batches.  Shape (batchSize, ?(dependent on loss function))
func (net *Network) Loss(ys, predictions *AF32, totalSamples int) float32 {
	switch net.LossFunction {
	case MeanSquaredError:
		return MeanSquaredErrorLoss(ys, predictions, totalSamples)
	case SparseCategoricalCrossEntropyFromLogits:
		return SparseCategoricalCrossEntropyLoss(ys, predictions, totalSamples)
	default:
		panic("unimplemented loss function type")
	}
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
	// This is garbage -- save scalars as {1} tensors
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
	return int(tensor.At1(0)), nil
}

func loadFloat32FromTensor(tensors map[string]*AF32, key string) (float32, error) {
	tensor, ok := tensors[key]
	if !ok {
		return 0, fmt.Errorf("missing tensor %s", key)
	}

	return tensor.At1(0), nil
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
		aep.wTranspose[l] = AF32Copy(net.Layers[l].W)
		aep.dadz[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.dadzTranspose[l] = AF32Copy(aep.dadz[l])
		aep.a[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.aTranspose[l] = AF32Copy(aep.a[l])
		aep.djda[l] = MakeAF32(batchSize, net.Layers[l].OutputSize)
		aep.djdaTranspose[l] = AF32Copy(aep.djda[l])
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
	batchSize := x.Shape[0]

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
	xTranspose := AF32Copy(x)
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
				djdw := aep.djdw[l].At2(i, j)
				oldmw := aep.oldMW[l].At2(i, j)
				oldvw := aep.oldVW[l].At2(i, j)
				aep.newMW[l].Set2(i, j, beta1*oldmw+(1-beta1)*djdw)
				aep.newVW[l].Set2(i, j, beta2*oldvw+(1-beta2)*djdw*djdw)
			}
		}
	}

	for l := 0; l < len(net.Layers); l++ {
		for i := 0; i < net.Layers[l].OutputSize; i++ {
			djdb := aep.djdb[l].At2(i, 0)
			oldmb := aep.oldMB[l].At2(i, 0)
			oldvb := aep.oldVB[l].At2(i, 0)
			aep.newMB[l].Set1(i, beta1*oldmb+(1-beta1)*djdb)
			aep.newVB[l].Set1(i, beta2*oldvb+(1-beta2)*djdb*djdb)
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
				newW := net.Layers[l].W.At2(i, j) - alphaT*aep.newMW[l].At2(i, j)/(math32.Sqrt(aep.newVW[l].At2(i, j))+aep.epsilon)
				net.Layers[l].W.Set2(i, j, newW)
			}

			newB := net.Layers[l].B.At1(i) - alphaT*aep.newMB[l].At1(i)/(math32.Sqrt(aep.newVB[l].At1(i))+aep.epsilon)
			net.Layers[l].B.Set1(i, newB)
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
	B *AF32 // Shape (OutputSize)

	InputSize  int
	OutputSize int
}

func MakeDense(activation ActivationType, inputSize, outputSize int, r *rand.Rand) *Layer {
	l := &Layer{
		Activation: activation,
		InputSize:  inputSize,
		OutputSize: outputSize,
		W:          MakeAF32(outputSize, inputSize),
		B:          MakeAF32(outputSize),
	}

	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			l.W.Set2(i, j, float32(r.NormFloat64())*0.1)
		}
		l.B.Set1(i, 0.1)
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
	batchSize := x.Shape[0]
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	if x.Shape[0] != batchSize {
		panic("dimension mismatch")
	}
	if x.Shape[1] != inputSize {
		panic("dimension mismatch")
	}
	_ = x.At2(0, inputSize-1)
	_ = x.At2(batchSize-1, inputSize-1)

	if a.Shape[0] != batchSize {
		panic("dimension mismatch")
	}
	if a.Shape[1] != outputSize {
		panic("dimension mismatch")
	}
	_ = a.At2(batchSize-1, outputSize-1)

	if dadz != nil {
		if dadz.Shape[0] != batchSize {
			panic("dimension mismatch")
		}
		if dadz.Shape[1] != outputSize {
			panic("dimension mismatch")
		}
		_ = dadz.At2(batchSize-1, outputSize-1)
	}

	if lay.W.Shape[0] != outputSize {
		panic("dimension mismatch")
	}
	if lay.W.Shape[1] != inputSize {
		panic("dimension mismatch")
	}
	_ = lay.W.At2(outputSize-1, inputSize-1)

	if !slices.Equal(lay.B.Shape, []int{outputSize}) {
		panic(fmt.Sprintf("lay.B.Shape != {outputSize}"))
	}
	_ = lay.B.At1(outputSize - 1)

	// Write the linear activations into a.  Equivalent to
	//
	// for k := 0; k < batchSize; k++ {
	// 	for i := 0; i < outputSize; i++ {
	// 		var z float32
	// 		for j := 0; j < inputSize; j++ {
	// 			z += lay.W.At(i, j) * x.At(k, j)
	// 		}
	// 		z += lay.B.At(i, 0)
	// 		a.Set(k, i, z)
	// 	}
	// }
	for k := 0; k < batchSize; k++ {
		for i := 0; i < outputSize; i++ {
			z := denseDot2SIMD(lay.W.V[i*inputSize:i*inputSize+inputSize], x.V[k*inputSize:k*inputSize+inputSize])
			z += lay.B.At1(i)
			a.Set2(k, i, z)
		}
	}

	// Apply activation function to a elementwise.  Store activation gradients
	// in dadz if provided.
	switch lay.Activation {
	case ReLU:
		if dadz != nil {
			reluActivationGradient(a.V, dadz.V)
		}
		reluActivation(a.V)
	case Linear:
		if dadz != nil {
			linearActivationGradient(dadz.V)
		}
		// linear activation is a no-op
	case Sigmoid:
		if dadz != nil {
			sigmoidActivationGradient(a.V, dadz.V)
		}
		sigmoidActivation(a.V)
	default:
		panic("unhandled activation function")
	}
}

// xT (input) is the layer input.  Shape (lay.InputSize, batchSize)
// djdaT (input) is the gradient of the loss wrt a.  Shape (lay.OutputSize, batchSize)
// dadzT (input) is the gradient of a_ik wrt z_ik.  Shape (lay.OutputSize, batchSize)
// dJdw (output) is the gradient of the loss wrt lay.W.  Shape (lay.OutputSize, lay.InputSize)
func (lay *Layer) BackpropDjdw(xT, djdaT, dadzT, djdw *AF32) {
	batchSize := xT.Shape[1]
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

	for i := 0; i < outputSize; i++ {
		for j := 0; j < inputSize; j++ {
			grad := denseDot3SIMD(
				djdaT.V[i*batchSize:i*batchSize+batchSize],
				dadzT.V[i*batchSize:i*batchSize+batchSize],
				xT.V[j*batchSize:j*batchSize+batchSize],
			)
			djdw.Set2(i, j, grad)
		}
	}
}

// djdaT (input) is the gradient of the loss wrt a.  Shape (lay.OutputSize, batchSize)
// dadzT (input) is the gradient of a_ik wrt z_ik.  Shape (lay.OutputSize, batchSize)
// dJdb (output) is the gradient of the loss wrt lay.B.  Shape (lay.OutputSize, 1)
func (lay *Layer) BackpropDjdb(djdaT, dadzT, dJdb *AF32) {
	batchSize := djdaT.Shape[1]
	outputSize := lay.OutputSize

	// Compute gradient of loss with respect to biases.
	iBase := 0
	for i := 0; i < outputSize; i++ {
		grad := denseDot2SIMD(djdaT.V[iBase:iBase+batchSize], dadzT.V[iBase:iBase+batchSize])
		dJdb.Set1(i, grad)

		iBase += batchSize
	}
}

// dJda (input) is the gradient of the loss wrt a.  Shape (batchSize, lay.OutputSize)
// dadz (input) is the gradient of a_ik wrt z_ik.  Shape (batchSize, lay.OutputSize)
// wT (input) is the layer weights, tranposed.  Shape (inputSize, outputSize)
// dJdx (output) is the gradient of the loss wrt x.  Shape (batchSize, lay.InputSize)
func (lay *Layer) BackpropDjdx(djda, dadz, wT, djdx *AF32) {
	batchSize := djda.Shape[0]
	inputSize := lay.InputSize
	outputSize := lay.OutputSize

	// This function is equivalent to:
	//
	// for k := 0; k < batchSize; k++ {
	// 	for j := 0; j < inputSize; j++ {
	// 		var grad float32
	// 		for i := 0; i < outputSize; i++ {
	// 			grad += djda.At(k, i) * dadz.At(k, i) * wT.At(j, i)
	// 		}
	// 		djdx.Set(k, j, grad)
	// 	}
	// }

	for k := 0; k < batchSize; k++ {
		for j := 0; j < inputSize; j++ {
			grad := denseDot3SIMD(
				djda.V[k*outputSize:k*outputSize+outputSize],
				dadz.V[k*outputSize:k*outputSize+outputSize],
				wT.V[j*outputSize:j*outputSize+outputSize],
			)
			djdx.Set2(k, j, grad)
		}
	}
}

// z (input/output)
func reluActivation(z []float32) {
	var z0, z1, z2, z3, zeros simd.Float32x8
	for len(z) >= 32 {
		z3 = simd.LoadFloat32x8Slice(z[24:])
		z2 = simd.LoadFloat32x8Slice(z[16:])
		z1 = simd.LoadFloat32x8Slice(z[8:])
		z0 = simd.LoadFloat32x8Slice(z[:])

		z3.Max(zeros).StoreSlice(z[24:])
		z2.Max(zeros).StoreSlice(z[16:])
		z1.Max(zeros).StoreSlice(z[8:])
		z0.Max(zeros).StoreSlice(z[:])

		z = z[32:]
	}

	// Handle tail of less than 32 but more than 8 elements.
	for len(z) >= 8 {
		z0 = simd.LoadFloat32x8Slice(z)
		z0.Max(zeros).StoreSlice(z)
		z = z[8:]
	}

	// Handle final tail of less than 8 elements
	if len(z) > 0 {
		z0 = simd.LoadFloat32x8SlicePart(z)
		z0.Max(zeros).StoreSlicePart(z)
	}
}

// reluActivationGradient computes the derivative of the ReLU function.
//
// z (input) is the pre-activation linear output of a layer.
//
// dadz (output) is the derivative of ReLU(z)
func reluActivationGradient(z, dadz []float32) {
	if len(z) != len(dadz) {
		panic("len(z) != len(dadz)")
	}

	for i := 0; i < len(z); i++ {
		if z[i] <= 0 {
			dadz[i] = 0
		} else {
			dadz[i] = 1
		}
	}
}

func linearActivationGradient(dadz []float32) {
	for i := 0; i < len(dadz); i++ {
		dadz[i] = 1
	}
}

func sigmoidActivation(z []float32) {
	for i := 0; i < len(z); i++ {
		z[i] = 1 / (1 + math32.Exp(-z[i]))
	}
}

func sigmoidActivationGradient(z, dadz []float32) {
	for i := 0; i < len(z); i++ {
		tmp := math32.Exp(-z[i])
		dadz[i] = -tmp / (1 + tmp) / (1 + tmp)
	}
}
