// Command imagegen generates images with Krea 2 Turbo fully in pure Go.
//
// Usage:
//
//	imagegen -m ~/.cache/manifold/krea2-models -p "a red fox" -o out.png \
//	         [-w 512] [-h 512] [--steps 8] [--seed 42]
package main

import (
	"context"
	"flag"
	"fmt"
	"image/png"
	"os"
	"os/signal"
	"time"

	"github.com/intelligencedev/born/qwenimage"
)

func main() {
	modelDir := flag.String("m", "", "model directory (see qwenimage.New for layout)")
	prompt := flag.String("p", "", "text prompt")
	out := flag.String("o", "out.png", "output PNG path")
	width := flag.Int("w", 512, "image width (multiple of 16)")
	height := flag.Int("h", 512, "image height (multiple of 16)")
	steps := flag.Int("steps", 8, "diffusion steps")
	seed := flag.Uint64("seed", 42, "RNG seed")
	flag.Parse()

	if *modelDir == "" || *prompt == "" {
		flag.Usage()
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	pipe, err := qwenimage.New(*modelDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	start := time.Now()
	fmt.Printf("generating %dx%d, %d steps, seed %d\n", *width, *height, *steps, *seed)
	img, err := pipe.Generate(ctx, *prompt, qwenimage.Options{
		Width: *width, Height: *height, Steps: *steps, Seed: *seed,
		OnStep: func(step, total int, d time.Duration) {
			fmt.Printf("  step %d/%d  %s\n", step, total, d.Round(time.Second))
		},
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer func() { _ = f.Close() }()
	if err := png.Encode(f, img); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Printf("saved %s in %s\n", *out, time.Since(start).Round(time.Second))
}
