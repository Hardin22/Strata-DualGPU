// src/kernels/verify_batch_parity.cpp - the verify window's one-launch forms against the per-token launches they
// replace, bitwise.
//
//     build/verify_batch_parity
//
// 1. kv_append_q4_step_multi (STRATA_DF_KVAPP) against n kv_append_q4_step calls: the whole K and V pools and the
//    host copies (KV streaming), with a non-resident page among the window's cells.
// 2. native_gr_post_multi against n native_gr_post calls (the hyper-connection write before the PLE block).
// 3. native_ple_postops_batch_snap (STRATA_DF_PLE) against the per-token loop of the verify window: per token
//    native_ple_postops, ple_history_advance and a copy of the history (the commit's snapshot); the results, the final
//    history and every snapshot, for windows of 2..8 tokens.
// GPU, synthetic, no model.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// The set-up copies and fills run on the legacy default stream, which does NOT order work on the non-blocking stream
// the kernels run on (and cudaMemset / a pageable cudaMemcpy may return before the device has finished them): each
// one is completed here, or a kernel could read its inputs - or have its outputs overwritten - while they land.
template <typename T>
T* dalloc(size_t n, int fill = 0) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, fill, n * sizeof(T) + 64), "memset");
    ck(cudaDeviceSynchronize(), "memset sync");
    return p;
}

template <typename T>
std::vector<T> read(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "read");
    return h;
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload sync");
    return d;
}

void report(const char* what, bool ok) {
    std::printf("%-64s %s\n", what, ok ? "bitwise equal" : "DIFFERENT");
    if (!ok) ++g_fail;
}

// ---- 1. Q4_0 K/V append
void kv_append(cudaStream_t s) {
    k::QsaShapes sh = k::qsa_real_shapes();
    sh.page_size = 64;
    const int H = (int) sh.n_head_kv, D = (int) sh.head_dim, P = (int) sh.page_size, pages = 4;
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 1) % pages;
    table[2] = -1;   // a block that is not resident: only the host copy is written
    int32_t* d_table = upload(table);
    const size_t pool = (size_t) pages * H * P * k::kv_q4_bytes_per_head(D);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 2.f);
    for (int n : {2, 3, 4, 8}) {
        for (int base : {5, 60, 2 * 64 + 61}) {   // inside a page, across a page boundary, into the missing page
            std::vector<float> kv((size_t) n * H * D), vv(kv.size());
            for (auto& x : kv) x = nd(rng);
            for (auto& x : vv) x = nd(rng);
            float* d_k = upload(kv);
            float* d_v = upload(vv);
            std::vector<int32_t> steps((size_t) n * k::kStepCount);
            for (int t = 0; t < n; ++t) k::qsa_step_fill(steps.data() + (size_t) t * k::kStepCount, base + t, sh);
            int32_t* d_steps = upload(steps);
            uint8_t *pk[2], *pv[2], *hk[2], *hv[2];
            for (int i = 0; i < 2; ++i) {
                pk[i] = dalloc<uint8_t>(pool, 0x5a); pv[i] = dalloc<uint8_t>(pool, 0x5a);
                hk[i] = dalloc<uint8_t>(pool, 0x5a); hv[i] = dalloc<uint8_t>(pool, 0x5a);
            }
            k::KvHostPools host0{}, host1{};
            host0.k_q4 = hk[0]; host0.v_q4 = hv[0];
            host1.k_q4 = hk[1]; host1.v_q4 = hv[1];
            for (int t = 0; t < n; ++t)
                k::kv_append_q4_step(pk[0], pv[0], d_table, d_steps + (size_t) t * k::kStepCount, d_k + (size_t) t * H * D,
                                     d_v + (size_t) t * H * D, sh, s, &host0);
            k::kv_append_q4_step_multi(pk[1], pv[1], d_table, d_steps, d_k, d_v, n, sh, s, &host1);
            ck(cudaStreamSynchronize(s), "kv append");
            const bool ok = read(pk[0], pool) == read(pk[1], pool) && read(pv[0], pool) == read(pv[1], pool) &&
                            read(hk[0], pool) == read(hk[1], pool) && read(hv[0], pool) == read(hv[1], pool);
            char what[128];
            std::snprintf(what, sizeof what, "kv_append_q4_step_multi, %d tokens from cell %d", n, base);
            report(what, ok);
            for (int i = 0; i < 2; ++i) { cudaFree(pk[i]); cudaFree(pv[i]); cudaFree(hk[i]); cudaFree(hv[i]); }
            cudaFree(d_k); cudaFree(d_v); cudaFree(d_steps);
        }
    }
    cudaFree(d_table);
}

// ---- 2. the hyper-connection write
void gr_post(cudaStream_t s) {
    const int N = 2560, HC = 4;
    std::mt19937 rng(12);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n : {2, 4, 8}) {
        std::vector<float> r((size_t) n * HC * N), bo((size_t) n * N), inj((size_t) n * HC);
        for (auto& x : r) x = nd(rng);
        for (auto& x : bo) x = nd(rng);
        for (auto& x : inj) x = 3.f * nd(rng);
        float* r0 = upload(r);
        float* r1 = upload(r);
        float* d_bo = upload(bo);
        float* d_inj = upload(inj);
        for (int t = 0; t < n; ++t)
            k::native_gr_post(r0 + (size_t) t * HC * N, d_bo + (size_t) t * N, d_inj + (size_t) t * HC,
                              r0 + (size_t) t * HC * N, N, HC, s);
        k::native_gr_post_multi(r1, d_bo, d_inj, r1, N, HC, n, (long long) HC * N, N, HC, s);
        ck(cudaStreamSynchronize(s), "gr post");
        char what[128];
        std::snprintf(what, sizeof what, "native_gr_post_multi, %d tokens (in place)", n);
        report(what, read(r0, r.size()) == read(r1, r.size()));
        cudaFree(r0); cudaFree(r1); cudaFree(d_bo); cudaFree(d_inj);
    }
}

// ---- 3. the PLE post-ops of a window, with the commit's history snapshots
void ple(cudaStream_t s) {
    constexpr int N = k::NG_N_EMBD, D = k::NG_HC_DIM, HS = k::NG_HIST * k::NG_HC_DIM;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> nk(D), nq(D), nc(D);
    for (auto* v : {&nk, &nq, &nc})
        for (auto& x : *v) x = 1.f + 0.1f * nd(rng);
    std::vector<uint16_t> conv((size_t) 4 * D);
    for (auto& x : conv) x = k::f16_from_f32(0.3f * nd(rng));
    k::PleWeights w;
    w.norm_key = upload(nk);
    w.norm_query = upload(nq);
    w.norm_conv = upload(nc);
    w.conv1d_f16 = upload(conv);
    std::vector<float> hist0((size_t) HS);
    for (auto& x : hist0) x = nd(rng);
    for (int n : {2, 3, 4, 8}) {
        std::vector<float> key((size_t) n * D), hid((size_t) n * D), val((size_t) n * N);
        for (auto& x : key) x = 4.f * nd(rng);
        for (auto& x : hid) x = nd(rng);
        for (auto& x : val) x = nd(rng);
        // per token, as the verify window's loop: native_ple_postops (ple_block_projected's buffers), the history
        // advance and the snapshot copy
        float* key0 = upload(key);
        float* hid0 = upload(hid);
        float* val0 = upload(val);
        float* hist_a = upload(hist0);
        float* snap_a = dalloc<float>((size_t) n * HS);
        float *q = dalloc<float>(D), *nrm = dalloc<float>(D), *g = dalloc<float>(4), *gd = dalloc<float>(D);
        float* cv = dalloc<float>(D);
        for (int t = 0; t < n; ++t) {
            k::NativePlePostopsBuffers b{q, nrm, g, gd, nrm, cv, hid0 + (size_t) t * D};
            k::native_ple_postops(key0 + (size_t) t * D, hid0 + (size_t) t * D, val0 + (size_t) t * N, hist_a, w, b, s);
            k::ple_history_advance(hist_a, nrm, s);
            ck(cudaMemcpyAsync(snap_a + (size_t) t * HS, hist_a, (size_t) HS * 4, cudaMemcpyDeviceToDevice, s), "snap");
        }
        // the window's one pass
        float* key1 = upload(key);
        float* hid1 = upload(hid);
        float* val1 = upload(val);
        float* hist_b = upload(hist0);
        float* snap_b = dalloc<float>((size_t) n * HS);
        float *qn = dalloc<float>((size_t) n * D), *gated = dalloc<float>((size_t) n * D), *gate = dalloc<float>((size_t) n * 4);
        k::native_ple_postops_batch_snap(key1, hid1, val1, hist_b, w, qn, gated, gate, n, snap_b, s);
        ck(cudaStreamSynchronize(s), "ple");
        char what[128];
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: results", n);
        report(what, read(hid0, hid.size()) == read(hid1, hid.size()));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: history", n);
        report(what, read(hist_a, (size_t) HS) == read(hist_b, (size_t) HS));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: snapshots", n);
        report(what, read(snap_a, (size_t) n * HS) == read(snap_b, (size_t) n * HS));
        for (float* p : {key0, hid0, val0, hist_a, snap_a, q, nrm, g, gd, cv, key1, hid1, val1, hist_b, snap_b, qn, gated,
                         gate})
            cudaFree(p);
    }
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    std::printf("verify_batch_parity: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    kv_append(s);
    gr_post(s);
    ple(s);
    std::printf("verify_batch_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
