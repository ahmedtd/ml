package toolbox

// func TestAdam2DLinregLearnsKnownModel(t *testing.T) {
// 	m0 := float32(10)
// 	m1 := float32(5)
// 	b := float32(30)

// 	xs := []*AF32{}
// 	ys := []*AF32{}

// 	batchSize := 32
// 	for i := 0; i < 10000/batchSize; i++ {
// 		x, y := generate2DLinRegDataset(batchSize, m0, m1, b)

// 		xs = append(xs, x)
// 		ys = append(ys, y)
// 	}

// 	r := rand.New(rand.NewSource(12345))

// 	net := &Network{
// 		LossFunction: MeanSquaredError,
// 		Layers: []*Layer{
// 			MakeDense(Linear, 2, 1, r),
// 		},
// 	}

// 	aep := net.MakeAdamParameters(0.001, batchSize)

// 	for epoch := 0; epoch < 124; epoch++ {
// 		for batch := 0; batch < len(xs); batch++ {
// 			net.AdamStep(xs[batch], ys[batch], aep, 1)
// 		}

// 		// Shuffle batches so we present them in a different order in the next epoch.
// 		r.Shuffle(len(xs), func(i, j int) {
// 			xs[i], xs[j] = xs[j], xs[i]
// 			ys[i], ys[j] = ys[j], ys[i]
// 		})
// 	}

// 	t.Logf("toolkit m0=%v m1=%v b=%v loss=%v", net.Layers[0].W.At(0, 0), net.Layers[0].W.At(0, 1), net.Layers[0].B.At(0, 0), net.Loss(xs, ys))

// 	if math32.Abs(net.Layers[0].W.At(0, 0)-m0) > 0.01 {
// 		t.Errorf("Disagreement on m0 parameter; got %v, want %v", net.Layers[0].W.At(0, 0), m0)
// 	}

// 	if math32.Abs(net.Layers[0].W.At(0, 1)-m1) > 0.01 {
// 		t.Errorf("Disagreement on m1 parameter; got %v, want %v", net.Layers[0].W.At(0, 1), m1)
// 	}

// 	if math32.Abs(net.Layers[0].B.At(0, 0)-b) > 0.01 {
// 		t.Errorf("Disagreement on b parameter; got %v, want %v", net.Layers[0].B.At(0, 0), b)
// 	}
// }

// func generate2DLinRegDataset(batchSize int, m0, m1, b float32) (x, y *AF32) {
// 	r := rand.New(rand.NewSource(12345))

// 	x = MakeAF32(batchSize, 2)
// 	y = MakeAF32(batchSize, 1)

// 	for k := 0; k < batchSize; k++ {
// 		// Normalization is important --- if I multiply x1 * 1000, the loss is
// 		// huge and the model blows up with NaNs.
// 		x0 := r.Float32()
// 		x1 := r.Float32()
// 		y0 := m0*x0 + m1*x1 + b

// 		x.Set(k, 0, x0)
// 		x.Set(k, 1, x1)
// 		y.Set(k, 0, y0)
// 	}

// 	return x, y
// }
