#include "edge_llm/nn.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace edge_llm {

Tensor rms_norm(
    const Tensor& input,
    const Tensor& weight,
    float epsilon
) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument(
            "RMSNorm epsilon must be finite and greater than zero"
        );
    }

    if (weight.rank() != 1) {
        throw std::invalid_argument(
            "RMSNorm weight must be a rank-1 tensor"
        );
    }

    const std::size_t width = input.shape().back();

    if (weight.size() != width) {
        throw std::invalid_argument(
            "RMSNorm weight size must match the input's last dimension"
        );
    }

    Tensor output(input.shape());

    const float* input_data = input.data();
    const float* weight_data = weight.data();
    float* output_data = output.data();

    const std::size_t group_count =
        input.size() / width;

    for (
        std::size_t group = 0;
        group < group_count;
        ++group
    ) {
        const std::size_t offset = group * width;
        double square_sum = 0.0;

        for (
            std::size_t index = 0;
            index < width;
            ++index
        ) {
            const double value =
                static_cast<double>(input_data[offset + index]);

            square_sum += value * value;
        }

        const double mean_square =
            square_sum / static_cast<double>(width);

        const float inverse_rms =
            static_cast<float>(
                1.0 / std::sqrt(
                    mean_square +
                    static_cast<double>(epsilon)
                )
            );

        for (
            std::size_t index = 0;
            index < width;
            ++index
        ) {
            output_data[offset + index] =
                input_data[offset + index] *
                inverse_rms *
                weight_data[index];
        }
    }

    return output;
}

Tensor softmax_last_dim(
    const Tensor& input
) {
    const std::size_t width = input.shape().back();
    const std::size_t group_count =
        input.size() / width;

    Tensor output(input.shape());

    const float* input_data = input.data();
    float* output_data = output.data();

    for (
        std::size_t group = 0;
        group < group_count;
        ++group
    ) {
        const std::size_t offset = group * width;
        float maximum = input_data[offset];

        for (
            std::size_t index = 1;
            index < width;
            ++index
        ) {
            const float value =
                input_data[offset + index];

            if (value > maximum) {
                maximum = value;
            }
        }

        double exponential_sum = 0.0;

        for (
            std::size_t index = 0;
            index < width;
            ++index
        ) {
            const float shifted =
                input_data[offset + index] - maximum;

            const float exponential =
                static_cast<float>(
                    std::exp(static_cast<double>(shifted))
                );

            output_data[offset + index] = exponential;
            exponential_sum += exponential;
        }

        for (
            std::size_t index = 0;
            index < width;
            ++index
        ) {
            output_data[offset + index] /=
                static_cast<float>(exponential_sum);
        }
    }

    return output;
}

}  // namespace edge_llm