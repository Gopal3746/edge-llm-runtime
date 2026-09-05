#pragma once

#include "edge_llm/tensor.hpp"

namespace edge_llm {

[[nodiscard]] Tensor rms_norm(
    const Tensor& input,
    const Tensor& weight,
    float epsilon = 1.0e-5F
);

[[nodiscard]] Tensor softmax_last_dim(
    const Tensor& input
);

}  // namespace edge_llm