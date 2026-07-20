//go:build !darwin && !linux && !windows

package moonshine

import (
	"fmt"

	"github.com/intelligencedev/born/internal/tensor"
)

func preferredBackend() (tensor.Backend, func(), error) {
	return nil, nil, fmt.Errorf("webgpu is not supported on this platform")
}
