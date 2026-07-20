//go:build darwin || linux || windows

package moonshine

import (
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/intelligencedev/born/backend/webgpu"
	"github.com/intelligencedev/born/internal/tensor"
)

func preferredBackend() (tensor.Backend, func(), error) {
	backend, err := webgpu.New()
	if err != nil {
		return nil, nil, err
	}
	if info := backend.AdapterInfo(); info != nil && info.DeviceType == gputypes.DeviceTypeCPU {
		backend.Release()
		return nil, nil, fmt.Errorf("webgpu: only a software adapter is available")
	}
	return backend, backend.Release, nil
}
