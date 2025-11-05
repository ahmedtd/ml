package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"math/rand"
	"os"

	"github.com/ahmedtd/ml/toolbox"
	"github.com/chewxy/math32"
	"github.com/google/subcommands"

	_ "image/jpeg"
	_ "image/png"
)

type InferCommand struct {
	weightsFile string
	imageFile   string
}

var _ subcommands.Command = (*InferCommand)(nil)

func (*InferCommand) Name() string {
	return "infer"
}

func (*InferCommand) Synopsis() string {
	return "Infer using the model weights"
}

func (*InferCommand) Usage() string {
	return ``
}

func (c *InferCommand) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.weightsFile, "weights", "mnist.safetensors", "Path to the weights produced by the train command")
	f.StringVar(&c.imageFile, "image", "", "Path to the image to predict")
}

func (c *InferCommand) Execute(ctx context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	if err := c.executeErr(ctx); err != nil {
		log.Printf("Error: %v", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}

func (c *InferCommand) executeErr(ctx context.Context) error {
	r := rand.New(rand.NewSource(12345))

	net := &toolbox.Network{
		LossFunction: toolbox.SparseCategoricalCrossEntropyFromLogits,
		Layers: []*toolbox.Layer{
			toolbox.MakeDense(toolbox.ReLU, 28*28, 256, r),
			toolbox.MakeDense(toolbox.ReLU, 256, 256, r),
			toolbox.MakeDense(toolbox.Linear, 256, 10, r),
		},
	}

	if err := c.loadWeights(net); err != nil {
		return fmt.Errorf("while loading weights: %w", err)
	}

	x, err := c.loadImage()
	if err != nil {
		return fmt.Errorf("while loading image: %w", err)
	}

	pred := net.Apply(x)

	digit := 0
	score := math32.Inf(-1)
	for i := 0; i < 10; i++ {
		if pred.At2(0, i) > score {
			digit = i
			score = pred.At2(0, i)
		}
	}

	log.Printf("Prediction: %d", digit)
	return nil
}

func (c *InferCommand) loadImage() (*toolbox.AF32, error) {
	f, err := os.Open(c.imageFile)
	if err != nil {
		return nil, fmt.Errorf("while opening image file: %w", err)
	}
	defer f.Close()

	rawImg, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("while decoding image: %w", err)
	}

	rawBounds := rawImg.Bounds()

	out := toolbox.MakeAF32(1, 28*28)

	for y := rawBounds.Min.Y; y < rawBounds.Max.Y; y++ {
		for x := rawBounds.Min.X; x < rawBounds.Max.X; x++ {
			v := float32(color.GrayModel.Convert(rawImg.At(x, y)).(color.Gray).Y) / float32(256)
			out.Set2(0, y*28+x, v)
		}
	}

	return out, nil
}

func (c *InferCommand) loadWeights(net *toolbox.Network) error {
	f, err := os.Open(c.weightsFile)
	if err != nil {
		return fmt.Errorf("while opening weights file: %w", err)
	}
	defer f.Close()

	tensors, err := toolbox.ReadSafeTensors(f)
	if err != nil {
		return fmt.Errorf("while reading weight tensors: %w", err)
	}

	if err := net.LoadTensors(tensors); err != nil {
		return fmt.Errorf("while restoring network: %w", err)
	}

	return nil
}
