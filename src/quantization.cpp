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

    const float* weight_data = weight.data();

    for (
        std::size_t index = 0;
        index < weight.size();
        ++index
    ) {
        if (!std::isfinite(weight_data[index])) {
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

    const float* float_weight_data =
        float_weight.data();

    std::int8_t* quantized_weight_data =
        quantized_weights_.data();

    float* scale_data =
        scales_.data();

    for (
        std::size_t output = 0;
        output < output_features_;
        ++output
    ) {
        const float* float_weight_row =
            float_weight_data +
            output * input_features_;

        std::int8_t* quantized_weight_row =
            quantized_weight_data +
            output * input_features_;

        float maximum_absolute_value = 0.0F;

        for (
            std::size_t input = 0;
            input < input_features_;
            ++input
        ) {
            maximum_absolute_value =
                std::max(
                    maximum_absolute_value,
                    std::fabs(float_weight_row[input])
                );
        }

        const float scale =
            maximum_absolute_value == 0.0F
                ? 1.0F
                : maximum_absolute_value /
                    maximum_quantized_value;

        scale_data[output] = scale;

        for (
            std::size_t input = 0;
            input < input_features_;
            ++input
        ) {
            const float scaled_value =
                float_weight_row[input] / scale;

            const long rounded_value =
                std::lround(scaled_value);

            const long clamped_value =
                std::clamp(
                    rounded_value,
                    -127L,
                    127L
                );

            quantized_weight_row[input] =
                static_cast<std::int8_t>(
                    clamped_value
                );
        }
    }
}

void QuantizedLinear::validate_input(
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
}

void QuantizedLinear::validate_output(
    const Tensor& input,
    const Tensor& output
) const {
    if (
        output.rank() != 2 ||
        output.shape()[0] != input.shape()[0] ||
        output.shape()[1] != output_features_
    ) {
        throw std::invalid_argument(
            "Quantized linear output must have shape "
            "[rows, output features]"
        );
    }

    if (input.data() == output.data()) {
        throw std::invalid_argument(
            "Quantized linear input and output must not alias"
        );
    }
}

Tensor QuantizedLinear::forward(
    const Tensor& input
) const {
    validate_input(input);

    Tensor output({
        input.shape()[0],
        output_features_
    });

    forward_into(
        input,
        output
    );

    return output;
}

void QuantizedLinear::forward_into(
    const Tensor& input,
    Tensor& output
) const {
    validate_input(input);
    validate_output(input, output);

    const std::size_t row_count =
        input.shape()[0];

    const float* input_data =
        input.data();

    float* output_data =
        output.data();

    const std::int8_t* quantized_weight_data =
        quantized_weights_.data();

    const float* scale_data =
        scales_.data();

    for (
        std::size_t row = 0;
        row < row_count;
        ++row
    ) {
        const float* input_row =
            input_data + row * input_features_;

        float* output_row =
            output_data + row * output_features_;

        for (
            std::size_t output_feature = 0;
            output_feature < output_features_;
            ++output_feature
        ) {
            const std::int8_t* weight_row =
                quantized_weight_data +
                output_feature * input_features_;

            float dot_product = 0.0F;

            for (
                std::size_t input_feature = 0;
                input_feature < input_features_;
                ++input_feature
            ) {
                dot_product +=
                    input_row[input_feature] *
                    static_cast<float>(
                        weight_row[input_feature]
                    );
            }

            output_row[output_feature] =
                dot_product *
                scale_data[output_feature];
        }
    }
}

Tensor QuantizedLinear::dequantized_weight() const {
    Tensor weight({
        output_features_,
        input_features_
    });

    float* output_data =
        weight.data();

    const std::int8_t* quantized_weight_data =
        quantized_weights_.data();

    const float* scale_data =
        scales_.data();

    for (
        std::size_t output = 0;
        output < output_features_;
        ++output
    ) {
        float* output_row =
            output_data +
            output * input_features_;

        const std::int8_t* quantized_row =
            quantized_weight_data +
            output * input_features_;

        const float scale =
            scale_data[output];

        for (
            std::size_t input = 0;
            input < input_features_;
            ++input
        ) {
            output_row[input] =
                static_cast<float>(
                    quantized_row[input]
                ) * scale;
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
