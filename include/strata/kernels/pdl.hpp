// include/strata/kernels/pdl.hpp - densefuse: programmatic dependent launch (PDL) for the verify window's chain.
//
// On sm_90+ a kernel launched with cudaLaunchAttributeProgrammaticStreamSerialization may start while its
// predecessor in the stream is still running: everything before `pdl_wait()` overlaps the predecessor (and its
// launch latency), everything after it sees the predecessor's writes, exactly as an ordinary launch would.  The
// kernels that opt in only LOAD constant weights before `pdl_wait()` and write nothing, so the values computed are
// unchanged.  `pdl_trigger()` lets the successor launch early (it still waits for this grid to finish before it
// reads anything).  On older GPUs (the 4060 Ti, sm_89) both are no-ops and the launch is an ordinary one.
//
// Stream capture turns the attribute into a programmatic graph edge.  A launch only uses it when the caller says the
// stream's previous captured node is a kernel (`pdl_scope`), never after a memcpy, an event wait or a host node.
#pragma once

#include <cuda_runtime.h>

#include <utility>

namespace strata::kernels {

/// Whether launches on this thread may currently carry the PDL attribute (set by the verify window's recorder
/// around stretches of kernel-only work on a device that supports it).
bool& pdl_scope();

/// The current device supports PDL and STRATA_DF_PDL is not 0.
bool pdl_supported();

#if defined(__CUDACC__)
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

/// `kernel<<<grid, block, smem, stream>>>(args...)`, with the PDL attribute when `pdl_scope()` is set.
template <typename... KArgs, typename... Args>
inline cudaError_t launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                              Args&&... args) {
    if (!pdl_scope()) {
        kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
        return cudaGetLastError();
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}
#elif defined(__HIPCC__)
__device__ __forceinline__ void pdl_wait() {}

__device__ __forceinline__ void pdl_trigger() {}

template <typename... KArgs, typename... Args>
inline hipError_t launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, hipStream_t stream,
                             Args&&... args) {
    kernel<<<grid, block, smem, stream>>>(std::forward<Args>(args)...);
    return hipGetLastError();
}
#endif  // __CUDACC__ / __HIPCC__

}  // namespace strata::kernels
