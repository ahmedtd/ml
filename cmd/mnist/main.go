// Command mnist implements training an inference on the MNIST dataset.
//
// To train: `go run ./cmd/mnist train --data-file=cmd/mnist/data/mnist.npz`
//
// To infer: `go run ./cmd/mnist infer --weights=mnist-out.safetensors --image=cmd/mnist/data/five.png`
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"

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
	subcommands.Register(&InferCommand{}, "")

	flag.Parse()
	ctx := context.Background()
	os.Exit(int(subcommands.Execute(ctx)))
}

type TrainCommand struct {
	dataFile string

	fromCheckpointFile string
	outputWeightFile   string

	cpuProfileFile string
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
	f.StringVar(&c.fromCheckpointFile, "from-checkpoint", "", "Path to initial weights to load for training")
	f.StringVar(&c.outputWeightFile, "output-weight-file", "mnist-out.safetensors", "Path to save trained weights (safetensors format)")

	f.StringVar(&c.cpuProfileFile, "cpu-profile", "", "Write a CPU profile")
}

func (c *TrainCommand) Execute(ctx context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	if err := c.executeErr(ctx); err != nil {
		log.Printf("Error: %v", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}

func (c *TrainCommand) executeErr(ctx context.Context) error {
	if c.cpuProfileFile != "" {
		f, err := os.Create(c.cpuProfileFile)
		if err != nil {
			return fmt.Errorf("while creating CPU profile file: %w", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			return fmt.Errorf("while starting CPU profile: %w", err)
		}
		defer pprof.StopCPUProfile()
	}

	xTrain, yTrain, xTest, yTest, err := loadMNIST(c.dataFile)
	if err != nil {
		return fmt.Errorf("while loading MNIST data set: %w", err)
	}

	// Group training data into batches, discarding the last
	// partially-full batch.
	batchSize := 2048
	xs := []*toolbox.AF32{}
	ys := []*toolbox.AF32{}

	sliceStart := 0
	for sliceStart < xTrain.Shape0 {
		x := toolbox.MakeAF32(batchSize, xTrain.Shape1)
		y := toolbox.MakeAF32(batchSize, yTrain.Shape1)
		for k := 0; k < batchSize; k++ {
			for j := 0; j < xTrain.Shape1; j++ {
				x.Set(k, j, xTrain.At((sliceStart+k)%xTrain.Shape0, j))
			}
			for j := 0; j < yTrain.Shape1; j++ {
				y.Set(k, j, yTrain.At((sliceStart+k)%xTrain.Shape0, j))
			}
		}
		xs = append(xs, x)
		ys = append(ys, y)

		sliceStart += batchSize
	}

	log.Printf("Data loaded and batched into %d batches", len(xs))

	r := rand.New(rand.NewSource(12345))

	net := &toolbox.Network{
		LossFunction: toolbox.SparseCategoricalCrossEntropyFromLogits,
		Layers: []*toolbox.Layer{
			toolbox.MakeDense(toolbox.ReLU, 28*28, 256, r),
			toolbox.MakeDense(toolbox.ReLU, 256, 256, r),
			toolbox.MakeDense(toolbox.Linear, 256, 10, r),
		},
	}

	// TODO: Use 4 threads instead of 1 thread
	aep := net.MakeAdamParameters(0.01, batchSize)

	if c.fromCheckpointFile != "" {
		if err := c.loadCheckpoint(net, aep); err != nil {
			return fmt.Errorf("while loading initial checkpoint: %w", err)
		}
	}

	for epoch := 0; epoch < 5; epoch++ {
		for batch := 0; batch < len(xs); batch++ {
			net.AdamStep(xs[batch], ys[batch], aep, 1)
		}

		// Shuffle batches so we present them in a different order in the next epoch.
		r.Shuffle(len(xs), func(i, j int) {
			xs[i], xs[j] = xs[j], xs[i]
			ys[i], ys[j] = ys[j], ys[i]
		})

		if err := c.writeCheckpoint(net, aep); err != nil {
			return fmt.Errorf("while writing checkpoint: %w", err)
		}

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

		log.Printf("epoch %d training-loss=%f training-pct=%.1f testing-loss=%f testing-pct=%.1f",
			epoch,
			net.Loss(yTrain, trainPred, xTrain.Shape0),
			trainPercent,
			net.Loss(yTest, testPred, xTest.Shape0),
			testPercent,
		)
		log.Printf("epoch %d timings overall=%.1f forward=%.1f loss=%.1f backprop=%.1f momentvectors=%.1f weightupdate=%.1f",
			epoch,
			aep.Timings.Overall.Seconds(),
			aep.Timings.Forward.Seconds(),
			aep.Timings.Loss.Seconds(),
			aep.Timings.Backpropagation.Seconds(),
			aep.Timings.MomentVectors.Seconds(),
			aep.Timings.WeightUpdate.Seconds(),
		)
		aep.Timings.Reset()
	}

	return nil
}

func (c *TrainCommand) loadCheckpoint(net *toolbox.Network, aep *toolbox.AdamEvaluationParameters) error {
	f, err := os.Open(c.fromCheckpointFile)
	if err != nil {
		return fmt.Errorf("while opening checkpoint file: %w", err)
	}
	defer f.Close()

	tensors, err := toolbox.ReadSafeTensors(f)
	if err != nil {
		return fmt.Errorf("while reading checkpoint tensors: %w", err)
	}

	if err := net.LoadTensors(tensors); err != nil {
		return fmt.Errorf("while restoring network: %w", err)
	}
	if err := aep.LoadTensors(tensors); err != nil {
		return fmt.Errorf("while restoring Adam: %w", err)
	}

	return nil
}

func (c *TrainCommand) writeCheckpoint(net *toolbox.Network, aep *toolbox.AdamEvaluationParameters) error {
	f, err := os.Create(c.outputWeightFile)
	if err != nil {
		return fmt.Errorf("while creating checkpoint file: %w", err)
	}
	defer f.Close()

	tensors := map[string]*toolbox.AF32{}

	net.DumpTensors(tensors)
	aep.DumpTensors(tensors)

	if err := toolbox.WriteSafeTensors(f, tensors); err != nil {
		return fmt.Errorf("while writing checkpoint tensors: %w", err)
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
		result.V[i] = float32(raw[i]) / float32(255)
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
