#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// With `layer_hi >= 0`, only the blk.<l>. matrices of layers [layer_lo, layer_hi) are uploaded (a layer split's
    /// stage: the other layers run on another card).
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1);
    /// Frees the uploaded matrices and the scratch, so `load` can run again.  The WeightRefs that pointed at them
    /// must not be used afterwards (the caller reloads their table too).
    void release();
    /// The uploaded bytes of layers [lo, hi).
    uint64_t bytes_in(int64_t lo, int64_t hi) const;
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size(); }

private:
    std::vector<void*> weights_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    std::vector<uint64_t> layer_bytes_;   ///< uploaded bytes per layer
};
} // namespace strata::core
