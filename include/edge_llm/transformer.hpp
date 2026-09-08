#pragma once

#include <cstddef>

#include "edge_llm/attention.hpp"
#include "edge_llm/feed_forward.hpp"
#include "edge_llm/tensor.hpp"

namespace edge_llm {

class TransformerBlock {
public:
    TransformerBlock(
        CausalSelfAttention attention,
        SwiGLUFeedForward feed_forward,
        Tensor attention_norm_weight,
        Tensor feed_forward_norm_weight,
        float norm_epsilon = 1.0e-5F
    );

    [[nodiscard]] Tensor forward(
        const Tensor& input,
        std::size_t position_offset = 0
    ) const;

    [[nodiscard]] std::size_t model_dimension() const noexcept;
    [[nodiscard]] std::size_t hidden_dimension() const noexcept;

private:
    CausalSelfAttention attention_;
    SwiGLUFeedForward feed_forward_;
    Tensor attention_norm_weight_;
    Tensor feed_forward_norm_weight_;
    float norm_epsilon_;
};

}  // namespace edge_llm