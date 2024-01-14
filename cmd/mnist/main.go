package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/ahmedtd/ml/toolbox"
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
	xTrain, yTrain, _, _, err := loadMNIST(c.dataFile)
	if err != nil {
		return fmt.Errorf("while loading MNIST data set: %w", err)
	}

	net := &toolbox.Network{
		LossFunction: toolbox.SparseCategoricalCrossEntropyFromLogits,
		Layers: []*toolbox.Layer{
			toolbox.MakeDense(toolbox.ReLU, 28*28, 25),
			toolbox.MakeDense(toolbox.ReLU, 25, 15),
			toolbox.MakeDense(toolbox.ReLU, 15, 10),
		},
	}

	net.GradientDescent()

}

func loadMNIST(path string) ([]byte, []byte, []byte, []byte, error) {
	r, err := npz.Open(path)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while opening mnist data file: %w", err)
	}
	defer r.Close()

	// It seems like even though the npy format supports specifying a Fortran
	// layout, numpy will always write C-style layouts (row-major / last index
	// stored contiguously.)

	// The MNIST data set is of 28x28 images.

	var xTrain []byte
	if err := r.Read("x_train.npy", &xTrain); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading x_train.npy: %w", err)
	}

	var yTrain []byte
	if err := r.Read("y_train.npy", &yTrain); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading y_train.npy: %w", err)
	}

	var xTest []byte
	if err := r.Read("x_test.npy", &xTest); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading x_test.npy: %w", err)
	}

	var yTest []byte
	if err := r.Read("y_test.npy", &yTest); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("while reading y_test.npy: %w", err)
	}

	return xTrain, yTrain, xTest, yTest, nil
}
