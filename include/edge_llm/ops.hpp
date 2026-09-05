#pragma once

#include "edge_llm/tensor.hpp"

namespace edge_llm {

[[nodiscard]] Tensor add(
    const Tensor& left,
    const Tensor& right
);

[[nodiscard]] Tensor multiply(
    const Tensor& left,
    const Tensor& right
);

[[nodiscard]] Tensor matmul(
    const Tensor& left,
    const Tensor& right
);

}  // namespace edge_llm