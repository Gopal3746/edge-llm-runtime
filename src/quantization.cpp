#include "edge_llm/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace edge_llm {
namespace {

constexpr float maximum_quantized_value = 127.0F;

void validate_float_weight(
    const Tensor& weight
) {
    if (weight.rank() != 2) {
        throw std::invalid_argument(
            "Quantized linear weight must have shape "
            "[output features, input features]"
        );
    }

    for (
        std::size_t index = 0;
        index < weight.size();
        ++index
    ) {
        if (!std::isfinite(weight.at(index))) {
            throw std::invalid_argument(
                "Quantized linear weight values must be finite"
            );
        }
    }
}

}  // namespace

QuantizedLinear::QuantizedLinear(
    const Tensor& float_weight
)
    : input_features_(0),
      output_features_(0),
      quantized_weights_(),
      scales_() {
    validate_float_weight(float_weight);

    output_features_ = float_weight.shape()[0];
    input_features_ = float_weight.shape()[1];

    quantized_weights_.resize(
        float_weight.size()
    );

    scales_.resize(
        output_features_
    );

    for (
        std::size_t output = 0;
        output < output_features_;
        ++output
    ) {
        float maximum_absolute_value = 0.0F;

        for (
            std::size_t input = 0;
            input < input_features_;
            ++input
        ) {
            const float value =
                float_weight.at(
                    output * input_features_ + input
                );

            maximum_absolute_value =
                std::max(
                    maximum_absolute_value,
                    std::fabs(value)
                );
        }

        const float scale =
            maximum_absolute_value == 0.0F
                ? 1.0F
                : maximum_absolute_value /
                    maximum_quantized_value;

        scales_[output] = scale;

        for (
            std::size_t input = 0;
            input < input_features_;
            ++input
        ) {
            const std::size_t offset =
                output * input_features_ + input;

            const float scaled_value =
                float_weight.at(offset) / scale;

            const long rounded_value =
                std::lround(scaled_value);

            const long clamped_value =
                std::clamp(
                    rounded_value,
                    -127L,
                    127L
                );

            quantized_weights_[offset] =
                static_cast<std::int8_t>(
                    clamped_value
                );
        }
    }
}

Tensor QuantizedLinear::forward(
    const Tensor& input
) const {
    if (
        input.rank() != 2 ||
        input.shape()[1] != input_features_
    ) {
        throw std::invalid_argument(
            "Quantized linear input must have shape "
            "[rows, input features]"
        );
    }

    const std::size_t row_count =
        input.shape()[0];

    Tensor output({
        row_count,
        output_features_
    });

    for (
        std::size_t row = 0;
        row < row_count;
        ++row
    ) {
        for (
            std::size_t output_feature = 0;
            output_feature < output_features_;
            ++output_feature
        ) {
            float accumulator = 0.0F;

            const float scale =
                scales_[output_feature];

            for (
                std::size_t input_feature = 0;
                input_feature < input_features_;
                ++input_feature
            ) {
                const std::size_t input_offset =
                    row * input_features_ +
                    input_feature;

                const std::size_t weight_index =
                    weight_offset(
                        output_feature,
                        input_feature
                    );

                const float dequantized_weight =
                    static_cast<float>(
                        quantized_weights_[weight_index]
                    ) * scale;

                accumulator +=
                    input.at(input_offset) *
                    dequantized_weight;
            }

            output.at(
                row * output_features_ +
                output_feature
            ) = accumulator;
        }
    }

    return output;
}

Tensor QuantizedLinear::dequantized_weight() const {
    Tensor weight({
        output_features_,
        input_features_
    });

    for (
        std::size_t output = 0;
        output < output_features_;
        ++output
    ) {
        for (
            std::size_t input = 0;
            input < input_features_;
            ++input
        ) {
            const std::size_t offset =
                weight_offset(output, input);

            weight.at(offset) =
                static_cast<float>(
                    quantized_weights_[offset]
                ) * scales_[output];
        }
    }

    return weight;
}

std::size_t
QuantizedLinear::weight_offset(
    std::size_t output_row,
    std::size_t input_column
) const {
    if (
        output_row >= output_features_ ||
        input_column >= input_features_
    ) {
        throw std::out_of_range(
            "Quantized weight index is out of range"
        );
    }

    return output_row * input_features_ +
        input_column;
}

std::size_t
QuantizedLinear::input_features() const noexcept {
    return input_features_;
}

std::size_t
QuantizedLinear::output_features() const noexcept {
    return output_features_;
}

std::size_t
QuantizedLinear::float_weight_bytes() const noexcept {
    return
        quantized_weights_.size() *
        sizeof(float);
}

std::size_t
QuantizedLinear::quantized_weight_bytes() const noexcept {
    return
        quantized_weights_.size() *
            sizeof(std::int8_t) +
        scales_.size() *
            sizeof(float);
}

double
QuantizedLinear::compression_ratio() const noexcept {
    return static_cast<double>(
        float_weight_bytes()
    ) / static_cast<double>(
        quantized_weight_bytes()
    );
}

float QuantizedLinear::scale_at(
    std::size_t output_row
) const {
    if (output_row >= output_features_) {
        throw std::out_of_range(
            "Quantization scale index is out of range"
        );
    }

    return scales_[output_row];
}

std::int8_t
QuantizedLinear::quantized_weight_at(
    std::size_t output_row,
    std::size_t input_column
) const {
    return quantized_weights_[
        weight_offset(
            output_row,
            input_column
        )
    ];
}

}  // namespace edge_llm
