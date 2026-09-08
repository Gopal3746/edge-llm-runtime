#pragma once

#include <cstddef>

#include "edge_llm/layers.hpp"
#include "edge_llm/tensor.hpp"

namespace edge_llm {

class SwiGLUFeedForward {
public:
    SwiGLUFeedForward(
        Tensor gate_weight,
        Tensor up_weight,
        Tensor down_weight
    );

    [[nodiscard]] Tensor forward(
        const Tensor& input
    ) const;

    [[nodiscard]] std::size_t model_dimension() const noexcept;
    [[nodiscard]] std::size_t hidden_dimension() const noexcept;

private:
    Linear gate_projection_;
    Linear up_projection_;
    Linear down_projection_;
};

}  // namespace edge_llm