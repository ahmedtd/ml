package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/ahmedtd/ml/toolbox"
	"github.com/chewxy/math32"
	"github.com/google/subcommands"
	"github.com/sbinet/npyio/npz"
)

func main() {
	subcommands.Register(subcommands.HelpCommand(), "")
	subcommands.Register(subcommands.FlagsCommand(), "")
	subcommands.Register(subcommands.CommandsCommand(), "")

	subcommands.Register(&TrainCommand{}, "")

	flag.Parse()
	ctx := context.Background()
	os.Exit(int(subcommands.Execute(ctx)))
}

type TrainCommand struct {
	dataFile string
}

var _ subcommands.Command = (*TrainCommand)(nil)

func (*TrainCommand) Name() string {
	return "train"
}

func (*TrainCommand) Synopsis() string {
	return "Train the model"
}

func (*TrainCommand) Usage() string {
	return ``
}

func (c *TrainCommand) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.dataFile, "data-file", "mnist.npz", "Path to the mnist.npz input file")
}

func (c *TrainCommand) Execute(ctx context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	if err := c.executeErr(ctx); err != nil {
		log.Printf("Error: %v", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}

func (c *TrainCommand) executeErr(ctx context.Context) error {
	xTrain, yTrain, xTest, yTest, err := loadMNIST(c.dataFile)
	if err != nil {
		return fmt.Errorf("while loading MNIST data set: %w", err)
	}

	// Group training data into batches of 32, discarding the last
	// partially-full batch.
	batchSize := 32
	xs := []*toolbox.AF32{}
	ys := []*toolbox.AF32{}
	for batch := 0; batch < xTrain.Shape0/batchSize; batch++ {
		sliceStart := batch * batchSize

		x := toolbox.MakeAF32(batchSize, xTrain.Shape1)
		y := toolbox.MakeAF32(batchSize, yTrain.Shape1)
		for k := 0; k < batchSize; k++ {
			for j := 0; j < xTrain.Shape1; j++ {
				x.Set(k, j, xTrain.At(k+sliceStart, j))
			}
			for j := 0; j < yTrain.Shape1; j++ {
				y.Set(k, j, yTrain.At(k+sliceStart, j))
			}
		}
		xs = append(xs, x)
		ys = append(ys, y)
	}

	log.Printf("Data loaded and batched")

	net := &toolbox.Network{
		LossFunction: toolbox.SparseCategoricalCrossEntropyFromLogits,
		Layers: []*toolbox.Layer{
			toolbox.MakeDense(toolbox.ReLU, 28*28, 25),
			toolbox.MakeDense(toolbox.ReLU, 25, 15),
			toolbox.MakeDense(toolbox.Linear, 15, 10),
		},
	}

	aep := net.MakeAdamParameters(0.001, batchSize, 4, 4)

	r := rand.New(rand.NewSource(12345))
	for epoch := 0; epoch < 100; epoch++ {
		for batch := 0; batch < len(xs); batch++ {
			net.AdamStep(xs[batch], ys[batch], aep)
		}

		// Shuffle batches so we present them in a different order in the next epoch.
		r.Shuffle(len(xs), func(i, j int) {
			xs[i], xs[j] = xs[j], xs[i]
			ys[i], ys[j] = ys[j], ys[i]
		})

		// Check performance on the train data set
		trainPred := net.Apply(xTrain)
		numCorrectTrain := 0
		for k := 0; k < trainPred.Shape0; k++ {
			digit := 0
			score := math32.Inf(-1)
			for i := 0; i < 10; i++ {
				if trainPred.At(k, i) > score {
					digit = i
					score = trainPred.At(k, i)
				}
			}

			if float32(digit) == yTrain.At(k, 0) {
				numCorrectTrain++
			}
		}
		trainPercent := float32(numCorrectTrain) / float32(yTrain.Shape0) * float32(100)

		// Check performance on the test data set.
		testPred := net.Apply(xTest)
		numCorrectTest := 0
		for k := 0; k < testPred.Shape0; k++ {
			digit := 0
			score := math32.Inf(-1)
			for i := 0; i < 10; i++ {
				if testPred.At(k, i) > score {
					digit = i
					score = testPred.At(k, i)
				}
			}

			if float32(digit) == yTest.At(k, 0) {
				numCorrectTest++
			}
		}
		testPercent := float32(numCorrectTest) / float32(yTest.Shape0) * float32(100)

		log.Printf("epoch %d training-loss=%f training-pct=%.1f testing-loss=%f testing-pct=%.1f", epoch, net.Loss(xs, ys), trainPercent, net.Loss([]*toolbox.AF32{xTest}, []*toolbox.AF32{yTest}), testPercent)
	}

	return nil
}

func loadMNIST(path string) (xTrain, yTrain, xTest, yTest *toolbox.AF32, err error) {
	r, err := npz.Open(path)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while opening mnist data file: %w", err)
	}
	defer r.Close()

	// It seems like even though the npy format supports specifying a Fortran
	// layout, numpy will always write C-style layouts (row-major / last index
	// stored contiguously.)

	// The MNIST data set is of 28x28 images.  We will return them all in an
	// array of shape (batchSize, 28*28).

	xTrain, err = loadImages(r, "x_train.npy")
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading x_train.npy: %w", err)
	}

	yTrain, err = loadLabels(r, "y_train.npy")
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading y_train.npy: %w", err)
	}

	xTest, err = loadImages(r, "x_test.npy")
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading x_test.npy: %w", err)
	}

	yTest, err = loadLabels(r, "y_test.npy")
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading y_test.npy: %w", err)
	}

	return xTrain, yTrain, xTest, yTest, nil
}

func loadImages(r *npz.Reader, name string) (*toolbox.AF32, error) {
	header := r.Header(name)

	var raw []uint8
	if err := r.Read(name, &raw); err != nil {
		return nil, fmt.Errorf("while reading uint8 array")
	}

	result := toolbox.MakeAF32(header.Descr.Shape[0], header.Descr.Shape[1]*header.Descr.Shape[2])
	for i := 0; i < len(raw); i++ {
		result.V[i] = float32(raw[i])
	}

	return result, nil
}

func loadLabels(r *npz.Reader, name string) (*toolbox.AF32, error) {
	header := r.Header(name)

	var raw []uint8
	if err := r.Read(name, &raw); err != nil {
		return nil, fmt.Errorf("while reading uint8 array")
	}

	result := toolbox.MakeAF32(header.Descr.Shape[0], 1)
	for i := 0; i < len(raw); i++ {
		result.V[i] = float32(raw[i])
	}

	return result, nil
}
