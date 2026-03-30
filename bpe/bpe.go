package bpe

import (
	_ "embed"
	"encoding/json"
	"log"
	"slices"
	"strings"
	"unicode"

	"github.com/dlclark/regexp2"
)

var (
	//go:embed gpt2.encoder.json
	gpt2EncoderJSON []byte

	//go:embed gpt2.vocab.bpe
	gpt2VocabBPE []byte

	GPT2PatternRegexp = regexp2.MustCompile(`<[|]endoftext[|]>|'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, regexp2.None)
)

var (
	gpt2SpecialPretokenize = [][]rune{
		[]rune("'s"),
		[]rune("'t"),
		[]rune("'re"),
		[]rune("'ve"),
		[]rune("'m"),
		[]rune("'ll"),
		[]rune("'d"),
	}
)

func SpecialTokensChunker(input string, specialTokens []string) (chunk, rest string, special bool) {
	minPos := len(input)
	minTok := ""
	for _, specialTok := range specialTokens {
		if ind := strings.Index(input, specialTok); ind != -1 && ind < minPos {
			minPos = ind
			minTok = specialTok
		}
	}

	if minPos == 0 {
		return minTok, input[len(minTok):], true
	} else {
		return input[:minPos], input[minPos:], false
	}
}

func GPT2Chunker(input []rune) (chunk, rest []rune) {
	if len(input) == 0 {
		return nil, nil
	}

	for _, special := range gpt2SpecialPretokenize {
		if len(input) >= len(special) {
			prefixStr := string(input[:len(special)])
			if prefixStr == string(special) {
				return input[:len(special)], input[len(special):]
			}
		}
	}

	// optional space followed by one or more unicode.Letter
	if chunk, rest := gpt2TakeSpaceAndLetters(input, unicode.IsLetter); chunk != nil {
		return chunk, rest
	}

	// optional space followed by one or more unicode.Number
	if chunk, rest := gpt2TakeSpaceAndLetters(input, unicode.IsNumber); chunk != nil {
		return chunk, rest
	}

	// optional space followed by one or more (not space, unicode.Letter, unicode.Number)
	if chunk, rest := gpt2TakeSpaceAndLetters(input, func(r rune) bool {
		return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}); chunk != nil {
		return chunk, rest
	}

	// one or more space followed by one not-space
	if chunk, rest := gpt2TakeOneOrMoreFollowedByNot(input, unicode.IsSpace, func(r rune) bool { return !unicode.IsSpace(r) }); chunk != nil {
		return chunk, rest
	}

	// one or more space
	if chunk, rest := gpt2TakeOneOrMore(input, unicode.IsSpace); chunk != nil {
		return chunk, rest
	}

	return input[:1], input[1:]
}

func gpt2TakeSpaceAndLetters(input []rune, pred func(rune) bool) (chunk, rest []rune) {
	if len(input) == 0 {
		return nil, nil
	}
	lim := 0
	if len(input) > lim && input[lim] == ' ' {
		lim++
	}
	if len(input) > lim && pred(input[lim]) {
		lim++
	} else {
		return nil, nil
	}
	for {
		if len(input) > lim && pred(input[lim]) {
			lim++
		} else {
			break
		}
	}

	return input[:lim], input[lim:]
}

func gpt2TakeOneOrMore(input []rune, pred func(rune) bool) (chunk, rest []rune) {
	lim := 0
	for {
		if len(input) > lim && pred(input[lim]) {
			lim++
		} else {
			break
		}
	}
	if lim >= 1 {
		return input[:lim], input[lim:]
	}
	return nil, nil
}

func gpt2TakeOneOrMoreFollowedByNot(input []rune, pred func(rune) bool, npred func(rune) bool) (chunk, rest []rune) {
	lim := 0
	for {
		if len(input) > lim && pred(input[lim]) {
			lim++
		} else {
			break
		}
	}
	if lim >= 1 && len(input) > lim && !npred(input[lim]) {
		return input[:lim], input[lim:]
	}
	return nil, nil
}

func GPT2TokenizerTable(specialTokens []string) map[string]int {
	tokenTable := map[string]int{}
	if err := json.Unmarshal(gpt2EncoderJSON, &tokenTable); err != nil {
		panic(err)
	}
	return tokenTable

	// // The GPT-2 tokenizer initializes its tokens with this swizzled byte list.
	// // All the printable ASCII characters (except space) are pulled to the
	// // front.
	// //
	// // I don't understand why this was necessary?  It seems accidental.  Why not
	// // just reflect this in the bpe file?  Why reorder them?
	// var gpt2SingleByteTokens []rune
	// for i := 33; i <= 126; i++ {
	// 	gpt2SingleByteTokens = append(gpt2SingleByteTokens, rune(i))
	// }
	// for i := 0; i <= 32; i++ {
	// 	gpt2SingleByteTokens = append(gpt2SingleByteTokens, rune(i+0x100))
	// }
	// for i := 127; i <= 255; i++ {
	// 	gpt2SingleByteTokens = append(gpt2SingleByteTokens, rune(i+0x100))
	// }

	// vocabBPELines := bytes.Split(gpt2VocabBPE, []byte("\n"))
	// // The first line is a comment.
	// vocabBPELines = vocabBPELines[1:]

	// curToken := 0

	// bpeRanks := map[string]int{}

	// // Put all the single-byte tokens into the map.
	// for _, runeVal := range gpt2SingleByteTokens {
	// 	bpeRanks[string([]rune{runeVal})] = curToken
	// 	curToken++
	// }

	// // Now put all the merged tokens into the map.
	// for _, line := range vocabBPELines {
	// 	if len(line) == 0 {
	// 		continue
	// 	}
	// 	leaves := bytes.Split(line, []byte(" "))
	// 	bpeRanks[string(leaves[0])+string(leaves[1])] = curToken
	// 	curToken++
	// }

	// for _, specialTok := range specialTokens {
	// 	switch specialTok {
	// 	case "<|endoftext|>":
	// 		bpeRanks["<|endoftext|>"] = 50256
	// 	default:
	// 		panic("unsupported special token")
	// 	}
	// }

	// return bpeRanks
}

func GPT2Tokenize(input string, specialTokens []string, mergeTable map[string]int) []int {
	// Adapted from TikToken educational: https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py

	//pretokenizedWords := pretokenizer.FindAllString(input, -1)

	// var pretokenizedWords []string
	// match, err := pretokenizer.FindStringMatch(input)
	// if err != nil {
	// 	panic(err)
	// }
	// pretokenizedWords = append(pretokenizedWords, match.Group.String())
	// for match != nil {
	// 	match, err = pretokenizer.FindNextMatch(match)
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	if match != nil {
	// 		pretokenizedWords = append(pretokenizedWords, match.Group.String())
	// 	}
	// }

	var tokens []int

	rest := input
	for {
		var chunk string
		var special bool
		chunk, rest, special = SpecialTokensChunker(rest, specialTokens)
		if chunk == "" {
			// No more input
			return tokens
		}

		log.Printf("chunk=%q special=%v", chunk, special)

		// If chunk was a special token, convert it and append it tokens.
		if special {
			log.Printf("token=%d", mergeTable[chunk])
			tokens = append(tokens, mergeTable[chunk])
			continue
		}

		// Otherwise, this chunk should be word-chunked and then each word-chunk
		// tokenized.  I'm not sure why GPT-2 includes the word-chunking step.
		// It seems unnecessary.
		chunkRunes := []rune(chunk)
		var wordChunkRunes [][]rune
		for {
			var wordChunk []rune
			wordChunk, chunkRunes = GPT2Chunker(chunkRunes)
			if wordChunk == nil {
				break
			}
			wordChunkRunes = append(wordChunkRunes, wordChunk)
		}

		var wordChunkStrings []string
		for _, chunk := range wordChunkRunes {
			wordChunkStrings = append(wordChunkStrings, string(chunk))
		}
		log.Printf("wordChunked=%q", wordChunkStrings)

		for _, wordChunk := range wordChunkRunes {

			// The GPT-2 tokenizer adds 0x100 to any unicode character that is an
			// ASCII control character or whitespace.  It's difficult for me to
			// imagine what breakage or misunderstanding of character encodings made
			// them feel this was necessary.
			for i := range wordChunk {
				if wordChunk[i] <= 32 || wordChunk[i] >= 127 {
					wordChunk[i] = wordChunk[i] + 0x100
				}
			}

			tokens = append(tokens, BPEBreak(wordChunk, mergeTable)...)
		}

	}
}

func BPEBreak(input []rune, mergeTable map[string]int) []int {
	// Break the string down into bytes.
	var mergeables []string
	for i := range len(input) {
		mergeables = append(mergeables, string(input[i:i+1]))
	}

	for {
		// Try to pick a pair of tokens to merge.
		minIndex := len(mergeables)
		minTok := len(mergeTable)
		minTokBytes := ""
		for i := 0; i < len(mergeables)-1; i++ {
			a := mergeables[i]
			b := mergeables[i+1]

			mergeTok, ok := mergeTable[a+b]
			if ok && mergeTok < minTok {
				minIndex = i
				minTok = mergeTok
				minTokBytes = a + b
			}
		}

		// If we found no valid merge, we are done.
		if minIndex == len(mergeables) {
			break
		}

		log.Printf("mergeables=%q merging=[%d:%d] replace=%v", mergeables, minIndex, minIndex+1, minTokBytes)

		mergeables[minIndex] = minTokBytes
		mergeables = slices.Delete(mergeables, minIndex+1, minIndex+2)
	}

	// Translate mergeables to tokens.
	var tokens []int
	for _, seq := range mergeables {
		tokens = append(tokens, mergeTable[seq])
	}

	log.Printf("original=%q mergeables=%q tokens=%v", input, mergeables, tokens)

	return tokens
}
