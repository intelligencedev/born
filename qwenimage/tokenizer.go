package qwenimage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer implements the Qwen2/Qwen3 byte-level BPE encoder from an HF
// tokenizer.json (model.type == "BPE", ByteLevel pre-tokenizer, NFC
// normalizer). Only Encode is needed for text conditioning.
type Tokenizer struct {
	vocab    map[string]int32
	merges   map[[2]string]int // pair -> rank
	specials []string          // added tokens, longest-first
	special  map[string]int32
}

type hfTokenizerJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Model struct {
		Type   string           `json:"type"`
		Vocab  map[string]int32 `json:"vocab"`
		Merges []string         `json:"merges"`
	} `json:"model"`
}

// LoadTokenizer reads tokenizer.json from dir (or a direct file path).
func LoadTokenizer(path string) (*Tokenizer, error) {
	if st, err := os.Stat(path); err == nil && st.IsDir() {
		path = filepath.Join(path, "tokenizer.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer: %w", err)
	}
	var tj hfTokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}
	if tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("unsupported tokenizer model type %q", tj.Model.Type)
	}
	t := &Tokenizer{
		vocab:   tj.Model.Vocab,
		merges:  make(map[[2]string]int, len(tj.Model.Merges)),
		special: make(map[string]int32, len(tj.AddedTokens)),
	}
	for rank, m := range tj.Model.Merges {
		left, right, ok := strings.Cut(m, " ")
		if !ok {
			return nil, fmt.Errorf("malformed merge %q", m)
		}
		t.merges[[2]string{left, right}] = rank
	}
	for _, at := range tj.AddedTokens {
		t.special[at.Content] = at.ID
		t.specials = append(t.specials, at.Content)
	}
	sort.Slice(t.specials, func(i, j int) bool { return len(t.specials[i]) > len(t.specials[j]) })
	return t, nil
}

// Encode tokenizes text without adding BOS/EOS (matches transformers
// add_special_tokens=False; special tokens appearing literally in the text,
// e.g. <|im_start|>, are still mapped to their ids).
func (t *Tokenizer) Encode(text string) []int32 {
	var ids []int32
	for _, seg := range t.splitSpecials(norm.NFC.String(text)) {
		if id, ok := t.special[seg]; ok {
			ids = append(ids, id)
			continue
		}
		for _, pre := range pretokenize(seg) {
			ids = append(ids, t.bpe(pre)...)
		}
	}
	return ids
}

// splitSpecials splits text on added special tokens (longest-first, leftmost).
func (t *Tokenizer) splitSpecials(text string) []string {
	var out []string
	for len(text) > 0 {
		bestIdx, bestTok := -1, ""
		for _, sp := range t.specials {
			if i := strings.Index(text, sp); i >= 0 && (bestIdx < 0 || i < bestIdx) {
				bestIdx, bestTok = i, sp
				if i == 0 {
					break
				}
			}
		}
		if bestIdx < 0 {
			out = append(out, text)
			break
		}
		if bestIdx > 0 {
			out = append(out, text[:bestIdx])
		}
		out = append(out, bestTok)
		text = text[bestIdx+len(bestTok):]
	}
	return out
}

// pretokenize emulates the Qwen2 HF Split regex:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//	 ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// Alternatives are tried in order at each position (leftmost-first). Go's RE2
// cannot express the (?!\S) lookahead, so this is a hand-rolled scanner,
// validated exactly against transformers via the tokenizer-cases fixture.
func pretokenize(text string) []string {
	rs := []rune(text)
	var out []string
	for i := 0; i < len(rs); {
		// 1: contractions (case-insensitive).
		if rs[i] == '\'' && i+1 < len(rs) {
			if n := matchContraction(rs[i:]); n > 0 {
				out = append(out, string(rs[i:i+n]))
				i += n
				continue
			}
		}
		// 2: optional single non-letter/digit/newline prefix + letters.
		if j := i; !isLetter(rs[j]) && rs[j] != '\r' && rs[j] != '\n' && !isNumber(rs[j]) && j+1 < len(rs) && isLetter(rs[j+1]) {
			k := j + 1
			for k < len(rs) && isLetter(rs[k]) {
				k++
			}
			out = append(out, string(rs[i:k]))
			i = k
			continue
		}
		if isLetter(rs[i]) {
			k := i
			for k < len(rs) && isLetter(rs[k]) {
				k++
			}
			out = append(out, string(rs[i:k]))
			i = k
			continue
		}
		// 3: single digit.
		if isNumber(rs[i]) {
			out = append(out, string(rs[i]))
			i++
			continue
		}
		// 4: optional space + punctuation run + trailing newlines.
		{
			j := i
			if rs[j] == ' ' && j+1 < len(rs) && isPunct(rs[j+1]) {
				j++
			}
			if isPunct(rs[j]) {
				k := j
				for k < len(rs) && isPunct(rs[k]) {
					k++
				}
				for k < len(rs) && (rs[k] == '\r' || rs[k] == '\n') {
					k++
				}
				out = append(out, string(rs[i:k]))
				i = k
				continue
			}
		}
		// 5: \s*[\r\n]+ — whitespace run that ends in newline(s).
		if unicode.IsSpace(rs[i]) {
			k := i
			for k < len(rs) && unicode.IsSpace(rs[k]) {
				k++
			}
			last := k - 1
			for last >= i && rs[last] != '\r' && rs[last] != '\n' {
				last--
			}
			if last >= i {
				// Run i..last+1 matches \s*[\r\n]+ greedily up to the final
				// newline within the whitespace run.
				out = append(out, string(rs[i:last+1]))
				i = last + 1
				continue
			}
			// 6: \s+(?!\S) — all-but-last whitespace when followed by
			// non-space; else the whole run.
			if k < len(rs) && k-i > 1 {
				out = append(out, string(rs[i:k-1]))
				i = k - 1
				continue
			}
			if k == len(rs) {
				out = append(out, string(rs[i:k]))
				i = k
				continue
			}
			// 7: single whitespace before non-space.
			out = append(out, string(rs[i]))
			i++
			continue
		}
		// Fallback (unreachable for valid input): emit rune as-is.
		out = append(out, string(rs[i]))
		i++
	}
	return out
}

func matchContraction(rs []rune) int {
	low := func(r rune) rune { return unicode.ToLower(r) }
	if len(rs) >= 2 {
		switch low(rs[1]) {
		case 's', 't', 'm', 'd':
			return 2
		}
	}
	if len(rs) >= 3 {
		two := string([]rune{low(rs[1]), low(rs[2])})
		switch two {
		case "re", "ve", "ll":
			return 3
		}
	}
	return 0
}

func isLetter(r rune) bool { return unicode.IsLetter(r) }
func isNumber(r rune) bool { return unicode.IsNumber(r) }

// isPunct matches [^\s\p{L}\p{N}] — anything that is not whitespace, letter,
// or number.
func isPunct(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

// byteToUnicode is the GPT-2 byte<->unicode bijection used by ByteLevel BPE.
var byteToUnicode = func() [256]rune {
	var table [256]rune
	n := 0
	isPrintable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	for b := 0; b < 256; b++ {
		if isPrintable(b) {
			table[b] = rune(b)
		} else {
			table[b] = rune(256 + n)
			n++
		}
	}
	return table
}()

// bpe byte-level-encodes one pretoken and applies ranked merges.
func (t *Tokenizer) bpe(pre string) []int32 {
	// Bytes -> unicode symbol string, one symbol per byte.
	parts := make([]string, 0, len(pre))
	for i := 0; i < len(pre); i++ {
		parts = append(parts, string(byteToUnicode[pre[i]]))
	}
	for len(parts) > 1 {
		bestRank, bestIdx := -1, -1
		for i := 0; i < len(parts)-1; i++ {
			if rank, ok := t.merges[[2]string{parts[i], parts[i+1]}]; ok {
				if bestRank < 0 || rank < bestRank {
					bestRank, bestIdx = rank, i
				}
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := parts[bestIdx] + parts[bestIdx+1]
		parts = append(parts[:bestIdx], append([]string{merged}, parts[bestIdx+2:]...)...)
	}
	ids := make([]int32, 0, len(parts))
	for _, p := range parts {
		id, ok := t.vocab[p]
		if !ok {
			// Strict: a missing vocab entry indicates a port bug, not data.
			panic(fmt.Sprintf("qwenimage tokenizer: symbol %q not in vocab (pretoken %q)", p, pre))
		}
		ids = append(ids, id)
	}
	return ids
}
