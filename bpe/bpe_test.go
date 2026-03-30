package bpe

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

var (
	text1       = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
	wantTokens1 = []int{15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13}
)

func TestAgainstTiktoken(t *testing.T) {
	gpt2Merges := GPT2TokenizerTable([]string{"<|endoftext|>"})

	gotTokens := GPT2Tokenize(text1, []string{"<|endoftext|>"}, gpt2Merges)

	if diff := cmp.Diff(gotTokens, wantTokens1); diff != "" {
		t.Fatalf("Bad output; diff (-got +want)\n%s", diff)
	}
}
