#pragma once

#include <cstddef>

#include "edge_llm/kv_cache.hpp"
#include "edge_llm/layers.hpp"
#include "edge_llm/tensor.hpp"

namespace edge_llm {

class CausalSelfAttention {
public:
    CausalSelfAttention(
        Tensor query_weight,
        Tensor key_weight,
        Tensor value_weight,
        Tensor output_weight,
        std::size_t head_count,
        double rope_base = 10000.0
    );

    [[nodiscard]] Tensor forward(
        const Tensor& input,
        std::size_t position_offset = 0
    ) const;

    [[nodiscard]] Tensor prefill(
        const Tensor& input,
        KVCache& cache
    ) const;

    [[nodiscard]] Tensor decode(
        const Tensor& input,
        KVCache& cache
    ) const;

    [[nodiscard]] std::size_t model_dimension() const noexcept;
    [[nodiscard]] std::size_t head_count() const noexcept;
    [[nodiscard]] std::size_t head_dimension() const noexcept;

private:
    void validate_input(const Tensor& input) const;
    void validate_cache(const KVCache& cache) const;

    Linear query_projection_;
    Linear key_projection_;
    Linear value_projection_;
    Linear output_projection_;

    std::size_t head_count_;
    double rope_base_;
};

}  // namespace edge_llm
