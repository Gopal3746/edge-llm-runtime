#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "edge_llm/tensor.hpp"

namespace edge_llm {

class QuantizedLinear {
public:
    explicit QuantizedLinear(
        const Tensor& float_weight
    );

    [[nodiscard]] Tensor forward(
        const Tensor& input
    ) const;

    [[nodiscard]] Tensor dequantized_weight() const;

    [[nodiscard]] std::size_t input_features() const noexcept;
    [[nodiscard]] std::size_t output_features() const noexcept;

    [[nodiscard]] std::size_t float_weight_bytes() const noexcept;
    [[nodiscard]] std::size_t quantized_weight_bytes() const noexcept;

    [[nodiscard]] double compression_ratio() const noexcept;

    [[nodiscard]] float scale_at(
        std::size_t output_row
    ) const;

    [[nodiscard]] std::int8_t quantized_weight_at(
        std::size_t output_row,
        std::size_t input_column
    ) const;

private:
    [[nodiscard]] std::size_t weight_offset(
        std::size_t output_row,
        std::size_t input_column
    ) const;

    std::size_t input_features_;
    std::size_t output_features_;

    std::vector<std::int8_t> quantized_weights_;
    std::vector<float> scales_;
};

}  // namespace edge_llm
