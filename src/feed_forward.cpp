#include "edge_llm/feed_forward.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace edge_llm {
namespace {

float silu(float value) {
    float sigmoid = 0.0F;

    if (value >= 0.0F) {
        sigmoid =
            1.0F /
            (
                1.0F +
                static_cast<float>(
                    std::exp(
                        -static_cast<double>(value)
                    )
                )
            );
    } else {
        const float exponential =
            static_cast<float>(
                std::exp(
                    static_cast<double>(value)
                )
            );

        sigmoid =
            exponential / (1.0F + exponential);
    }

    return value * sigmoid;
}

}  // namespace

SwiGLUFeedForward::SwiGLUFeedForward(
    Tensor gate_weight,
    Tensor up_weight,
    Tensor down_weight
)
    : gate_projection_(std::move(gate_weight)),
      up_projection_(std::move(up_weight)),
      down_projection_(std::move(down_weight)) {
    const std::size_t model_dimension =
        gate_projection_.input_features();

    const std::size_t hidden_dimension =
        gate_projection_.output_features();

    if (
        up_projection_.input_features() != model_dimension ||
        up_projection_.output_features() != hidden_dimension ||
        down_projection_.input_features() != hidden_dimension ||
        down_projection_.output_features() != model_dimension
    ) {
        throw std::invalid_argument(
            "SwiGLU projection dimensions are incompatible"
        );
    }
}

Tensor SwiGLUFeedForward::forward(
    const Tensor& input
) const {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "SwiGLU input must have shape "
            "[sequence, model dimension]"
        );
    }

    if (input.shape()[1] != model_dimension()) {
        throw std::invalid_argument(
            "SwiGLU input width must match model dimension"
        );
    }

    const Tensor gate_values =
        gate_projection_.forward(input);

    const Tensor up_values =
        up_projection_.forward(input);

    Tensor gated_values(gate_values.shape());

    const float* gate_data = gate_values.data();
    const float* up_data = up_values.data();
    float* gated_data = gated_values.data();

    for (
        std::size_t index = 0;
        index < gated_values.size();
        ++index
    ) {
        gated_data[index] =
            silu(gate_data[index]) *
            up_data[index];
    }

    return down_projection_.forward(gated_values);
}

std::size_t
SwiGLUFeedForward::model_dimension() const noexcept {
    return gate_projection_.input_features();
}

std::size_t
SwiGLUFeedForward::hidden_dimension() const noexcept {
    return gate_projection_.output_features();
}

}  // namespace edge_llm