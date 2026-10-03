// include/strata/core/mtp.hpp - plan v0.3 P6: the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer (vLLM 0.30.0 `qwen4_exp` MTP; transcribed and validated in
// `tools/mtp_probe.py`, where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3).  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (`tools/mtp_rt.py` quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads `rt_dir` (from tools/mtp_rt.py) and allocates the layer's K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized: this takes ~0.9 GB.
    bool load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768);
    /// The prompt's length: prefill() skips the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// At most this many drafts per round (below max_t - 1): a window longer than the MTP's comes from elsewhere.
    void set_max_drafts(int k) { max_drafts_ = k; }
    int max_t() const { return max_t_; }
    uint64_t vram_bytes() const { return vram_; }
    /// The draft layer's K/V state (read-only: --serve's STRATA_STATE_HASH check hashes it)
    const QsaState& kv_state() const { return st_; }
    /// KV streaming: refill the ring of the drafter's window from its host copy for a sequence that continues at
    /// `upto` (a conversation-cache resume). No-op unless the drafter's K/V is a ring.
    void kv_restore(int64_t upto);
    /// The VRAM bind() will allocate for a native head of `head_row_bytes` per vocabulary row: the draft logits and
    /// the draft head over rt/draft_vocab.bin's subset.  The expert cache is sized before bind(), so it reserves this.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const;
    /// The main model's embedding and head, and the verify window's final residuals (T rows, hc*n_embd each).
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err);

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1).  Runs in batches of up to max_t rows.
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                 bool sync = true);
    /// --pipeline-windows: capture the prefill graphs and size the prefill records for windows of up to max_t rows
    /// now, so a later prefill(sync = false) neither captures nor allocates (both would wait for the device).
    bool prepare_prefill(std::string& err);
    cudaStream_t stream() const { return cs_; }

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);
    /// TRIMS: `draft` in two halves.  With the one-graph chain, `draft_begin` stages and launches the round and
    /// returns at once (the caller emits tokens meanwhile); `draft_end` waits and hands out what `draft` would have.
    /// Otherwise `draft_begin` runs the whole round and `draft_end` only copies its results.
    bool draft_begin(int T, const int32_t* tokens, int64_t p, int a, float min_p, std::string& err);
    bool draft_end(int32_t* drafts, float* probs, int* n_drafts, std::string& err);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    /// COUPLED DRAFT SAMPLING (core/coupled_draft.hpp; STRATA_SPEC_COUPLED=1, set up by bind()): the request's
    /// sampling.  A sampled request (temperature > 0, not greedy) then drafts by SAMPLING with the target's chain and
    /// the target's Philox draw for the verifying row, and `probs` is the draft's probability under that chain; a
    /// greedy one keeps the argmax drafts and their graphs.  A no-op when the switch is off.
    void set_draft_sampling(const strata::kernels::SamplerParams& sp);
    /// Coupled mode with penalties: the history the next draft() chain starts from - what the session holds after the
    /// window's commit (`tail`) and the window's pick at its last accepted row (`next`, the next window's row 0).
    void set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next);
    bool coupled() const { return coupled_active_; }

    /// STRATA_SHADOW_CHAIN (measurement only): after the drafts of each round, continue the chain to `n` outputs in
    /// total, whatever their probability; the extra outputs never reach a window (drafts and probs are unchanged), and
    /// the cells the extra steps write lie past the window, overwritten before anything reads them (see above).
    void set_shadow(int n) { shadow_total_ = n; }

    // ---- --pipeline-windows 2: the chain as one asynchronous launch (round + steps back to back, no host sync), its
    // first `n_force` steps fed the given tokens instead of their own picks (teacher forcing: the drafts of a window
    // already in flight), the outputs read once its event has completed.
    /// Before the first capture: the round/step graphs get the forcing kernel (a no-op while force[j] < 0).
    void set_force_capture(bool on) { force_on_ = on; }
    /// The rows the next round reads, copied into the bound buffer first (when they live elsewhere).
    void set_source_R(const float* rows) { src_R_ = rows; }
    /// Capture the round graphs for T = 1..max_t and every step graph now (a capture syncs the drafter's stream).
    bool prepare_chain(std::string& err);
    /// `n_early` (0: none): an extra event after that many outputs, polled by chain_early_poll.
    /// `n_headless`: steps 1..n_headless run without the head (their outputs read as -1): forced steps whose pick
    /// only serves as an agreement signal, ~0.15-0.2 ms each of the drafter's card.
    bool chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
                      int n_early, std::string& err, int n_headless = 0);
    /// 1: the outputs are in chain_tok()/chain_prob(); 0: still running; -1: error.
    int chain_poll(std::string& err);
    /// 1: the first `n_early` outputs are in chain_tok()/chain_prob() (the rest still running); 0: not yet; -1: error.
    int chain_early_poll(std::string& err);
    /// How many of the chain's first outputs (up to `n_early`; all of them once the chain is done) have landed:
    /// they are in chain_tok()/chain_prob().  -1: error.  A fresh window's size is often decided by its first outputs
    /// (a draft below spec_min_p ends it), so its stage-0 launch need not wait for all `n_early`.
    int chain_outputs_ready(std::string& err);
    bool chain_live() const { return chain_live_; }
    const int32_t* chain_tok() const { return chain_tok_; }
    const float* chain_prob() const { return chain_prob_; }
    cudaEvent_t chain_event() const { return ev_chain_; }
    int shadow_n() const { return shadow_n_; }
    const int32_t* shadow_tok() const { return shadow_tok_; }
    const float* shadow_prob() const { return shadow_prob_; }

    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;

    /// E-9: the prompt path computes this layer's prompt K/V in batches (Prefill::draft_kv): its tensors, its K/V
    /// state, the first cell a round can still read, its device, and a wait for its own stream.
    const float* tensor_f32(const char* name) const { return f32(name); }
    const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
    const void* tensor_q8(const char* name) const { return q8(name); }
    QsaState& kv_state_rw() { return st_; }
    int64_t first_needed() const { return (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0; }
    int device() const { return device_; }
    bool idle(std::string& err) {
        if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: its stream failed"; return false; }
        return true;
    }

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    bool capture_prefill(int T, std::string& err);
    bool capture_prefill_dev(int T, std::string& err);   ///< E-4: without the mapped staging (inputs copied on device)
    bool capture_round(int T, bool coupled, std::string& err);
    bool capture_step(int j, bool coupled, std::string& err);
    bool capture_step_forced(int j, std::string& err);   ///< a forced step without the head (its hidden, the token)
    cudaGraphExec_t step_exec_[9] = {};
    // TRIMS (STRATA_MTP_CHAIN, default on): the round and every chain step in ONE graph, step j behind a conditional
    // node that runs it only while drafts 0..j-1 all reached min_p (a mapped float the host sets per round) - the
    // host's own stopping rule evaluated on the GPU, so a round costs one launch and one wait instead of one per step.
    bool capture_chain(int T, std::string& err);
    bool draft_steps(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err, float* probs,
                     float min_p, int* n_drafts);   // the per-step rounds (no chain graph)
    int pend_mode_ = 0;                   // draft_begin: 1 = the chain graph is in flight, 2 = done (results below)
    int pend_jmax_ = 0;
    float pend_minp_ = 0.0f;
    int32_t pend_drafts_[16] = {};
    float pend_probs_[16] = {};
    int pend_n_ = 0;
    std::chrono::steady_clock::time_point pend_t0_{};
    cudaGraphExec_t chain_exec_[9] = {};
    bool chain_failed_ = false;           ///< a capture failed once: the per-step graphs from then on
    cudaStream_t cs2_ = nullptr;          ///< the conditional bodies are captured on it
    float *h_minp_ = nullptr, *m_minp_ = nullptr;
    cudaGraphExec_t step_forced_exec_[9] = {};
    bool skip_head_ = false;
    int chain_headless_ = 0;
    // coupled draft sampling: its own round/step graphs (the argmax ones stay as they were), the request's
    // parameters and the penalty ring (mapped staging + device copies), the split scratch, token id -> subset index
    bool setup_coupled(std::string& err);
    cudaGraphExec_t round_exec_c_[9] = {};
    cudaGraphExec_t step_exec_c_[9] = {};
    bool coupled_ok_ = false, coupled_active_ = false;
    bool coupled_rec_ = false;   ///< record_forward: the full layer ends in the coupled sampler (draft coupled_j_)
    int coupled_j_ = 0;
    strata::kernels::SamplerParams *h_cparams_ = nullptr, *m_cparams_ = nullptr, *cparams_ = nullptr;
    int32_t *h_chist_ = nullptr, *m_chist_ = nullptr, *cring_ = nullptr, *dinv_ = nullptr;
    void* cscratch_ = nullptr;
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0;
    int device_ = -1;   ///< the device `load` ran on: the public calls switch to it (layer split)
    int max_drafts_ = 1 << 30;
    int shadow_total_ = 0, shadow_n_ = 0;
    bool force_on_ = false, chain_live_ = false;
    const float* src_R_ = nullptr;
    int32_t *h_force_ = nullptr, *m_force_ = nullptr;
    cudaEvent_t ev_chain_ = nullptr, ev_early_ = nullptr;
    cudaEvent_t ev_step_[8] = {};      ///< after each of the first n_early outputs
    int steps_seen_ = 0;
    int chain_n_ = 0, chain_early_ = 0;
    bool early_live_ = false;
    int32_t chain_tok_[8] = {};
    float chain_prob_[8] = {};
    int32_t shadow_tok_[8] = {};
    float shadow_prob_[8] = {};
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t prefill_exec_[9] = {};
    cudaGraphExec_t prefill_dev_exec_[9] = {};
    int32_t* pf_dev_ = nullptr;   ///< E-4: a prompt's rows' token / step / position records, uploaded at once
    int64_t pf_cap_ = 0;          ///< its capacity in ints
    cudaGraphExec_t round_exec_[9] = {};

    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<Tensor> tensors_;
    uint8_t* dense_ = nullptr;
    uint8_t* experts_ = nullptr;
    void* state_arena_ = nullptr;
    QsaState st_;
    void* arena_ = nullptr;

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t prompt_len_ = 0;
    float* probs_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* xn_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t* hit_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    uint16_t* x_bf16_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
