package toolbox

import (
	"math/rand"
	"testing"

	"github.com/chewxy/math32"
)

func TestAdam1DLinregLearnsKnownModel(t *testing.T) {
	m := float32(10)
	b := float32(30)

	xs := []*AF32{}
	ys := []*AF32{}

	batchSize := 32
	for i := 0; i < 10000/batchSize; i++ {
		x, y := generate1DLinRegDataset(batchSize, m, b)

		xs = append(xs, x)
		ys = append(ys, y)
	}

	net := &Network{
		LossFunction: MeanSquaredError,
		Layers: []*Layer{
			MakeDense(Linear, 1, 1),
		},
	}

	aep := net.MakeAdamParameters(0.001, batchSize, 4, 1)

	r := rand.New(rand.NewSource(12345))
	for epoch := 0; epoch < 130; epoch++ {
		for batch := 0; batch < len(xs); batch++ {
			net.AdamStep(xs[batch], ys[batch], aep)
		}

		// Shuffle batches so we present them in a different order in the next epoch.
		r.Shuffle(len(xs), func(i, j int) {
			xs[i], xs[j] = xs[j], xs[i]
			ys[i], ys[j] = ys[j], ys[i]
		})
	}

	t.Logf("toolkit m=%v b=%v loss=%v", net.Layers[0].W.At(0, 0), net.Layers[0].B.At(0, 0), net.Loss(xs, ys))

	if math32.Abs(net.Layers[0].W.At(0, 0)-m) > 0.1 {
		t.Errorf("Disagreement on m parameter; got %v, want %v", net.Layers[0].W.At(0, 0), m)
	}

	if math32.Abs(net.Layers[0].B.At(0, 0)-b) > 0.1 {
		t.Errorf("Disagreement on b parameter; got %v, want %v", net.Layers[0].B.At(0, 0), b)
	}
}

func generate1DLinRegDataset(batchSize int, m, b float32) (x, y *AF32) {
	r := rand.New(rand.NewSource(12345))

	x = MakeAF32(batchSize, 1)
	y = MakeAF32(batchSize, 1)

	for i := 0; i < batchSize; i++ {
		// Normalization is important --- if I multiply x1 * 1000, the loss is
		// huge and the model blows up with NaNs.
		x1 := r.Float32()
		y1 := m*x1 + b

		// Perturb the point a little bit
		// y1 += 0.1*math32.Sin(0.001*x1) + (r.Float32()-0.5)*10

		x.Set(i, 0, x1)
		y.Set(i, 0, y1)
	}

	return x, y
}
