// src/kernels/cpu/iq_avx2.cpp - the i-quant expert rows in 256-bit lanes, several tokens at once.
//
// The AVX-512 kernels' (iq_avx512.cpp) multi-token scheme on the CPUs without AVX-512 (AMD Zen 2/3, Intel
// Core 12th-14th gen and Core Ultra): each 32-value chunk is decoded once - grid lookups, one sign vector,
// one scale vector - and every token of the verify window applies them with a load, a `sign`, a `maddubs`,
// a `madd` and an `add`.  ggml-cpu's own AVX2 dot products for these formats are single-token: every token
// re-does the codebook lookups and the sign expansion.  The sign vector is ggml's bit_selector pattern
// (ggml-cpu/arch/x86/quants.c): pshufb-broadcast of each sign byte, AND with the bit selector, CMPEQ, OR
// with one, `vpsignb`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references) - only the
// order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23).  IQ1_M stays on ggml-cpu.
#include "strata/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

void cpuid(int leaf, int sub, int r[4]) {
#if defined(_MSC_VER)
    __cpuidex(r, leaf, sub);
#else
    unsigned a, b, c, d;
    __cpuid_count((unsigned) leaf, (unsigned) sub, a, b, c, d);
    r[0] = (int) a; r[1] = (int) b; r[2] = (int) c; r[3] = (int) d;
#endif
}

// Whether the calling thread runs on a core whose vpgatherdd beats eight scalar loads and inserts: an Intel core,
// on a hybrid CPU only the performance cores (CPUID 1Ah core type 40h; the E-cores, 20h, gather slower than they
// insert).  AMD Zen 2/3 gather slowly, so they keep the inserts.  Asked once per thread: the pool pins its workers
// to one core each, and a CPUID can cost a microsecond under a hypervisor (Windows with VBS).  STRATA_IQ_GATHER=0/1
// forces it.
int cpu_vendor() {   // 0 = other, 1 = Intel non-hybrid, 2 = Intel hybrid
    static const int vendor = [] {
        int r[4];
        cpuid(0, 0, r);
        if (!(r[1] == 0x756e6547 && r[3] == 0x49656e69 && r[2] == 0x6c65746e)) return 0;   // "GenuineIntel"
        cpuid(7, 0, r);
        return ((r[3] >> 15) & 1) ? 2 : 1;
    }();
    return vendor;
}

bool gather_fast() {
    static const int forced = [] { const char* v = std::getenv("STRATA_IQ_GATHER"); return v ? std::atoi(v) : -1; }();
    if (forced >= 0) return forced != 0;
    const int vendor = cpu_vendor();
    if (vendor != 2) return vendor == 1;
    thread_local int on_p = -1;
    if (on_p < 0) {
        int r[4];
        cpuid(0x1A, 0, r);
        on_p = (((unsigned) r[0] >> 24) == 0x40) ? 1 : 0;
    }
    return on_p == 1;
}

// AVX-VNNI (Intel Alder Lake and later, P- and E-cores; CPUID 7.1:EAX[4]): vpdpwssd does a madd and its add in one
// uop, vpdpbusd a maddubs and a madd(ones).  Every kernel below has a VN variant that uses them; the integer sums are
// the same, so the results are bit for bit those of the plain AVX2 form.  MSVC emits the intrinsics under /arch:AVX2;
// GCC/Clang only with -mavxvnni, otherwise VN changes nothing.  STRATA_NO_AVXVNNI=1 turns them off.
#if defined(_MSC_VER) || defined(__AVXVNNI__)
#define STRATA_IQ_VNNI 1
#else
#define STRATA_IQ_VNNI 0
#endif
bool avx_vnni() {
    static const bool on = [] {
        if (const char* v = std::getenv("STRATA_NO_AVXVNNI"); v != nullptr && std::atoi(v) != 0) return false;
        int r[4];
        cpuid(7, 1, r);
        return STRATA_IQ_VNNI && ((r[0] >> 4) & 1) != 0;
    }();
    return on;
}
template <bool VN>
inline __m256i madd_add(__m256i acc, __m256i p16, __m256i sc) {   // acc + madd(p16, sc)
#if STRATA_IQ_VNNI
    if constexpr (VN) return _mm256_dpwssd_avx_epi32(acc, p16, sc);
#endif
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p16, sc));
}
template <bool VN>
inline __m256i dot4u(__m256i u8, __m256i s8) {   // four u8 x s8 products per int32 lane
#if STRATA_IQ_VNNI
    if constexpr (VN) return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u8, s8);
#endif
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// 32 sign bits -> 32 bytes of -1 (bit set) / +1: ggml's bit_selector pattern, shared by all tokens.
inline __m256i sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int) m);
    const __m128i lo = _mm_shuffle_epi8(bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// The two 16-value scales of one 32-value half (IQ2_XS, IQ2_S): int16 lanes 0-7 = a (values 0-15),
// 8-15 = b (values 16-31).  NOT a broadcast dword [a,b] - that would alternate the scales lane by lane.
inline __m256i sc16(int a, int b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)),
                                   _mm_set1_epi16((short) b), 1);
}
// One scale for the whole 32-value half (IQ2_XXS, IQ3_XXS, IQ3_S).
inline __m256i sc32(int s) { return _mm256_set1_epi16(s); }

// keven_signs_q2xs (ggml keeps it static in arch/x86/quants.c): one u64 per 7-bit sign index, byte k
// = 0xFF when bit k of ksigns_iq2xs[i] is set, 0x01 otherwise.  With this a whole 32-value sign vector
// is four scalar loads and a set_epi64x - no ksigns byte packing chain and no bit_selector expansion.
// Built once at load time from the shared ggml-common table so it can never drift from it.
struct EvenSigns {
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k)
                r |= (uint64_t) (((ksigns_iq2xs[i] >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static const EvenSigns even_signs;

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// E-2 (iq_avx512.cpp) on the AVX-2 path: the pool streams the expert rows from DRAM at ~25 GB/s (4 KB pages
// when large pages are refused), so ask for the bytes a few blocks before the decode needs them - 2048 B is
// two gate/up rows ahead, a row is ~1 KB.  Same switch as the AVX-512 kernels: STRATA_IQ_PREFETCH is the
// distance in bytes, 0 = off, default 2048.  Measured on a Zen 3 5700X3D (no AVX-512) on IQ3_S decode: -4% on
// the gate/up phase, +1.0 GB/s over the rows, -1.3% ms/round end to end.  The non-temporal hint measured
// worse than T0 at the same distance, so this keeps T0.
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

inline void rows_ahead(const uint8_t* p) {
    if (prefetch_ahead <= 0) return;
    _mm_prefetch((const char*) p, _MM_HINT_T0);
    _mm_prefetch((const char*) p + 64, _MM_HINT_T0);
}

// ---- per format: one 32-value half (values 64*j + 32*half .. +31) -> grid magnitudes, sign vector, scales
template <int TY> struct Fmt32;

template <> struct Fmt32<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t w0 = u32(q), w1 = u32(q + 4);
        g = _mm256_set_epi64x((long long) iq2xxs_grid[w0 >> 24], (long long) iq2xxs_grid[(w0 >> 16) & 255],
                              (long long) iq2xxs_grid[(w0 >> 8) & 255], (long long) iq2xxs_grid[w0 & 255]);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w1 >> 21) & 127], (long long) even_signs.v[(w1 >> 14) & 127],
                                (long long) even_signs.v[(w1 >> 7) & 127], (long long) even_signs.v[w1 & 127]);
        sc = sc32(2 * (int) (w1 >> 28) + 1);
    }
};

template <> struct Fmt32<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        uint16_t v[8];
        std::memcpy(v, b + 2 + 16 * j, 16);
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2xs_grid[v[o + 3] & 511], (long long) iq2xs_grid[v[o + 2] & 511],
                              (long long) iq2xs_grid[v[o + 1] & 511], (long long) iq2xs_grid[v[o] & 511]);
        uint32_t s = 0;
        for (int l = 0; l < 4; ++l) s |= (uint32_t) ksigns_iq2xs[v[o + l] >> 9] << (8 * l);
        sgn = sgn_vec(s);
        const uint8_t sb = b[66 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h = b[66 + 2 * j + half];
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2s_grid[qs[o + 3] | ((h << 2) & 0x300)],
                              (long long) iq2s_grid[qs[o + 2] | ((h << 4) & 0x300)],
                              (long long) iq2s_grid[qs[o + 1] | ((h << 6) & 0x300)],
                              (long long) iq2s_grid[qs[o] | ((h << 8) & 0x300)]);
        const uint64_t m = u64(b + 2 + 32 + 8 * j);
        sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
        const uint8_t sb = b[74 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127],
                                (long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

template <> struct Fmt32<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
#define G3(k) (int) iq3s_grid[q[k] | (((h >> k) & 1u) << 8)]
        g = _mm256_set_epi32(G3(7), G3(6), G3(5), G3(4), G3(3), G3(2), G3(1), G3(0));
#undef G3
        const uint64_t m = u64(b + 74 + 8 * j);
        sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
        const uint8_t s = b[106 + j];
        sc = sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
    }
};

// ---- the gathered decodes, for the cores where gather_fast() holds
// the 32 sign bits of a half, from the j's 8 sign bytes: one broadcast and one shuffle (sgn_vec: two and an insert)
inline __m256i sgn_half(const uint8_t* m8, int half) {
    const __m256i sb = _mm256_shuffle_epi8(
        _mm256_set1_epi64x((long long) u64(m8)),
        half ? _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7)
             : _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sb, sel), sel), _mm256_set1_epi8(1));
}
// four 7-bit even-sign indices of a u32 (bits 0-6, 7-13, 14-20, 21-27) -> their four masks, one gather
inline __m256i even_signs4(uint32_t w) {
    const __m128i idx = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) w), _mm_setr_epi32(0, 7, 14, 21)),
                                      _mm_set1_epi32(127));
    return _mm256_i32gather_epi64((const long long*) even_signs.v, idx, 8);
}

// IQ3_S on a core with fast gathers (an Intel P-core, see gather_fast): the same grid words and sign bytes as
// Fmt32<21>, so the same products bit for bit.  The eight 9-bit grid indices are built in one register (8 low bits
// from qs, the 9th from bit k of qh) and read with one gather; the scalar form spends ~7 shuffle-port uops per half
// on vpinsrd and runs the index arithmetic on the scalar ports.  On the 14900KF's P-cores: gate/up rows of an IQ3_S
// expert 0.41 -> 0.27 ms at three tokens; its E-cores gather slower than they insert, so they keep Fmt32<21>.
template <> struct Fmt32<121> {
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
        const __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q));
        const __m256i hb = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32((int) h),
                                                              _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1)),
                                            _mm256_set1_epi32(0x100));
        g = _mm256_i32gather_epi32((const int*) iq3s_grid, _mm256_or_si256(lo, hb), 4);
        sgn = sgn_half(b + 74 + 8 * j, half);
        const uint8_t s = b[106 + j];
        sc = sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
    }
};

// IQ3_XXS and IQ2_S the same way: their grid entries (and IQ3_XXS's even-sign masks) read with one gather from
// indices built in a register, and IQ2_S's explicit sign bits taken from one broadcast.  Each is its format's Fmt32
// bit for bit.  P-core gate/up rows at three tokens: IQ3_XXS 0.26 -> 0.24 ms, IQ2_S 0.31 -> 0.27.  IQ2_XXS measured
// 4% slower gathered (its four 8-byte grid loads are few enough already) and keeps Fmt32<16>.
template <> struct Fmt32<118> {   // IQ3_XXS, gathered
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_i32gather_epi32((const int*) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q)), 4);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = even_signs4(w);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

template <> struct Fmt32<122> {   // IQ2_S, gathered
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* qs = b + 2 + 8 * j + 4 * half;
        const uint32_t h = b[66 + 2 * j + half];
        // index k: qs byte k | bits 2k, 2k+1 of qh at bits 8-9
        const __m128i idx = _mm_or_si128(
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) u32(qs))),
            _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32((int) h), _mm_setr_epi32(8, 6, 4, 2)), _mm_set1_epi32(0x300)));
        g = _mm256_i32gather_epi64((const long long*) iq2s_grid, idx, 8);
        sgn = sgn_half(b + 2 + 32 + 8 * j, half);
        const uint8_t sb = b[74 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<23> {   // IQ4_XS: d, scales_h, scales_l[4], qs[128] - 136 B, 8 signed sub-scales
    // ggml's ggml_vec_dot_iq4_xs_q8_K: the same 16-value codebook as IQ4_NL, but each of the eight 32-value
    // sub-blocks carries its own 6-bit scale, read as two nibbles of scales_l[p] plus two bits of scales_h, and
    // used SIGNED as (ls - 32).  The scale therefore folds into the int16 operand of madd_epi16 instead of
    // becoming a float multiply per sub-block, and the codebook sign is carried the way iq4nl_rows does it.
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const int H = 2 * j + half;   // the eight 32-value sub-blocks, in the order the activation bytes come
        const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
        const __m128i bits = _mm_loadu_si128((const __m128i*) (b + 8 + 16 * H));
        const __m128i m4 = _mm_set1_epi8(0x0f);
        const __m256i q4 = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values, _mm_and_si128(bits, m4))),
            _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)), 1);
        g   = _mm256_sign_epi8(q4, q4);                       // |w|, the unsigned operand of maddubs
        sgn = _mm256_sign_epi8(_mm256_set1_epi8(1), q4);      // w's sign, applied to the activation
        const int ls = ((b[4 + (H >> 1)] >> (4 * (H & 1))) & 0xf) | (((u16(b + 2) >> (2 * H)) & 3) << 4);
        sc = _mm256_set1_epi16((short) (ls - 32));
    }
};

template <int TY, int NT, bool VN>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
        rows_ahead(blk + prefetch_ahead);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = madd_add<VN>(acci[t], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        const float dx = h2f(u16(blk)) * Fmt32<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// ---- IQ2_XS (17) with ggml's vectorized sign decode (arch/x86/quants.c, ggml_vec_dot_iq2_xs_q8_K):
// the 7-bit sign indices never touch ksigns_iq2xs - the 8th sign bit is reconstructed by parity (two
// shifts, a xor and ggml's bit_helper pshufb) and the per-half sign vectors come out of shuffles of one
// 32-byte load.  The 8 scale bytes become all 16 half-scales with ggml's unpack trick.  The grid lookups
// stay scalar loads into set_epi64x (ggml does the same).  Every token still only pays a load, a sign,
// a maddubs, a madd and an add per half, into alternating accumulators.
template <int NT, bool VN>
inline void row_dot_iq2xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const uint8_t bit_sel[32] = {
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80,
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80 };
    static const char shuf_even[32] = {   // sign bits of u16 0,1,2,3 (even bytes of the sign register)
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6 };
    static const char shuf_odd[32] = {    // sign bits of u16 4,5,6,7
        8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10,
        12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14 };
    static const uint8_t bit_help[32] = {  // 0x80 when the 4-bit index has odd popcount (parity bit 7)
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00,
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00 };
    static const uint8_t k_sc_shuffle[128] = {  // half h -> scale bytes 2h (lanes 0-7), 2h+1 (lanes 8-15)
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 };
    const __m256i bsel = _mm256_loadu_si256((const __m256i*) bit_sel);
    const __m256i shev = _mm256_loadu_si256((const __m256i*) shuf_even);
    const __m256i shod = _mm256_loadu_si256((const __m256i*) shuf_odd);
    const __m256i bhelp = _mm256_loadu_si256((const __m256i*) bit_help);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    const __m256i one8 = _mm256_set1_epi8(1);

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 74;
        rows_ahead(blk + prefetch_ahead);
        // the 8 scale bytes -> 16 half-scales of 2*s+1, interleaved [a0, b0, a1, b1, ...] (ggml's unpack)
        __m128i st = _mm_set1_epi64x((long long) u64(blk + 66));
        st = _mm_unpacklo_epi8(_mm_and_si128(st, m4), _mm_and_si128(_mm_srli_epi16(st, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(st, 1), m1);
        __m256i acc[NT][2];
        for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_si256();
        for (int jj = 0; jj < 2; ++jj) {   // 128 values (16 u16) per step
            const __m256i q2 = _mm256_loadu_si256((const __m256i*) (blk + 2 + 32 * jj));
            const __m256i p7 = _mm256_srli_epi16(q2, 9);              // 7 sign bits per u16, even bytes
            const __m256i p3 = _mm256_srli_epi16(q2, 13);             // sign index bits 4-6
            const __m256i fsb = _mm256_or_si256(p7,
                _mm256_shuffle_epi8(bhelp, _mm256_xor_si256(p7, p3)));  // + the parity sign bit 7
            const __m256i s01 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(fsb)); // u16 0-7
            const __m256i s23 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(fsb, 1)); // u16 8-15
            alignas(32) uint16_t gi[16];
            _mm256_store_si256((__m256i*) gi, _mm256_and_si256(q2, m511));
            for (int h = 0; h < 4; ++h) {   // the four 32-value halves of this group
                const int H = 4 * jj + h;   // global half index -> scales byte pair
                const __m256i g = _mm256_set_epi64x((long long) iq2xs_grid[gi[4 * h + 3]],
                                                   (long long) iq2xs_grid[gi[4 * h + 2]],
                                                   (long long) iq2xs_grid[gi[4 * h + 1]],
                                                   (long long) iq2xs_grid[gi[4 * h]]);
                const __m256i sb = _mm256_shuffle_epi8(h < 2 ? s01 : s23, (h & 1) ? shod : shev);
                const __m256i sgn = _mm256_or_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(sb, bsel), bsel), one8);
                const __m256i sc = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*) k_sc_shuffle + H)));
                const int off = 128 * jj + 32 * h;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc[t][h & 1] = madd_add<VN>(acc[t][h & 1], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        const float dx = h2f(u16(blk)) * 0.125f;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(acc[t][0], acc[t][1])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int TY, int NT, bool VN>
inline void row_dot_any(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    if constexpr (TY == 17) row_dot_iq2xs<NT, VN>(row, nblocks, y, res);
    else                    row_dot<TY, NT, VN>(row, nblocks, y, res);
}

template <int TY, int NT, bool VN>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT, VN>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot_any<TY, NT, VN>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT, bool VN>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT, VN>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY, bool VN>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8, VN>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY, bool VN>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1, VN>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2, VN>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3, VN>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4, VN>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY, VN>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// the VNNI choice for one format
template <int TY>
void gu_any(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
            int r0, int r1) {
    if (avx_vnni()) gu_rows_nt<TY, true>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
    else gu_rows_nt<TY, false>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
}
template <int TY>
void dot_any(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
             int r1) {
    if (avx_vnni()) dot_rows_nt<TY, true>(nt, w, row_bytes, n, act, out, r0, r1);
    else dot_rows_nt<TY, false>(nt, w, row_bytes, n, act, out, r0, r1);
}

}  // namespace

// ggml's quantize_row_q8_K_ref (ggml-quants.c), which is also what ggml-cpu runs on x86 (no SIMD version there),
// in AVX-2: the same per-value arithmetic - one IEEE multiply by the same iscale, the same 1.5*2^23 rounding add,
// the same clamp - so the bytes are identical (tools/cpu_expert_bench.cpp checks it).  The block's sign-carrying
// max is the FIRST value of the largest magnitude, as in the scalar loop; NaNs are skipped as there (`ax > amax`
// is false for them, and max_ps returns its second operand for a NaN first one).  A zero block leaves bsums as
// the reference does.  ~10x the scalar loop: a verify window quantizes up to 6 tokens per layer on the host before
// the pool can start.
void q8k_quant_avx2(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 magic = _mm256_set1_ps(12582912.f);
    const __m256i mant = _mm256_set1_epi32(0x007fffff), off = _mm256_set1_epi32(0x00400000);
    const __m256i c127 = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 8) m = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), m);
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const __m256 va = _mm256_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 8) {
            const int msk = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), va, _CMP_EQ_OQ));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m256 vis = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 32) {
            __m256i v[4];
            for (int q = 0; q < 4; ++q) {
                const __m256 t = _mm256_add_ps(_mm256_mul_ps(vis, _mm256_loadu_ps(x + j + 8 * q)), magic);
                v[q] = _mm256_min_epi32(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(t), mant), off), c127);
                // the reference stores MIN(127, v) into an int8 (keeps the low byte), and its bsums add those bytes:
                // the same here, so even a degenerate block (iscale overflowing to inf) gives the same bytes
                v[q] = _mm256_srai_epi32(_mm256_slli_epi32(v[q], 24), 24);
            }
            for (int q = 0; q < 2; ++q) {   // two 16-value sums
                const __m256i s = _mm256_add_epi32(v[2 * q], v[2 * q + 1]);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
                y[i].bsums[j / 16 + q] = (int16_t) _mm_cvtsi128_si32(s4);
            }
            const __m256i p16a = _mm256_packs_epi32(v[0], v[1]), p16b = _mm256_packs_epi32(v[2], v[3]);
            const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), perm);
            _mm256_storeu_si256((__m256i*) (y[i].qs + j), p8);
        }
        y[i].d = 1 / iscale;
    }
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

bool iq256_one_token(int type) noexcept {
    // IQ3_S on an Intel CPU: this kernel beats ggml's single-token dot at one token too (gathered on the P-cores,
    // 0.34 -> 0.19 ms per expert's gate/up rows on a 14900KF; 0.71 -> 0.68 on its E-cores).  Other CPUs not measured.
    return type == 21 && cpu_vendor() != 0;
}

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_any<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_any<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18:
            if (gather_fast()) gu_any<118>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
            else gu_any<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
            break;
        case 21:
            if (gather_fast()) gu_any<121>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
            else gu_any<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
            break;
        case 22:
            if (gather_fast()) gu_any<122>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
            else gu_any<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
            break;
        case 23: gu_any<23>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_any<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_any<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18:
            if (gather_fast()) dot_any<118>(nt, w, row_bytes, n, act, out, r0, r1);
            else dot_any<18>(nt, w, row_bytes, n, act, out, r0, r1);
            break;
        case 21:
            if (gather_fast()) dot_any<121>(nt, w, row_bytes, n, act, out, r0, r1);
            else dot_any<21>(nt, w, row_bytes, n, act, out, r0, r1);
            break;
        case 22:
            if (gather_fast()) dot_any<122>(nt, w, row_bytes, n, act, out, r0, r1);
            else dot_any<22>(nt, w, row_bytes, n, act, out, r0, r1);
            break;
        case 23: dot_any<23>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

// ---- IQ4_NL (type 20): 32-value blocks of 18 bytes (f16 d + 16 nibble bytes) against Q8_0 activations.
// ggml-cpu's own AVX-2 dot (ggml_vec_dot_iq4_nl_q8_0) is single-token: every token redoes the nibble ->
// kvalues_iq4nl pshufb decode and the |w| half of the signed x signed product.  Here both are computed once
// per block; every token then costs a sign, a maddubs, a madd and an fmadd.  The arithmetic is ggml's - only
// the order of the float additions differs.
template <int NT, bool VN>
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            rows_ahead(blk + prefetch_ahead);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                      // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);    // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);   // |w|: the unsigned operand of maddubs
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p = dot4u<VN>(aq, _mm256_sign_epi8(q8, q4));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(accf[t]);
    }
}

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    const bool vn = avx_vnni();
    switch (nt) {
        case 1: vn ? iq4nl_rows<1, true>(w, row_bytes, n, y, out, r0, r1) : iq4nl_rows<1, false>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: vn ? iq4nl_rows<2, true>(w, row_bytes, n, y, out, r0, r1) : iq4nl_rows<2, false>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: vn ? iq4nl_rows<3, true>(w, row_bytes, n, y, out, r0, r1) : iq4nl_rows<3, false>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: vn ? iq4nl_rows<4, true>(w, row_bytes, n, y, out, r0, r1) : iq4nl_rows<4, false>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl256_down_rows(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
        }
    }
}

}  // namespace strata::kernels::cpu
