#pragma once

#include <cstddef>

#include "edge_llm/tensor.hpp"

namespace edge_llm {

[[nodiscard]] Tensor apply_rope(
    const Tensor& input,
    std::size_t position_offset = 0,
    double base = 10000.0
);

}  // namespace edge_llm