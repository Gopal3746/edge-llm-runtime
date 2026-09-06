#include "edge_llm/rope.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace edge_llm {

Tensor apply_rope(
    const Tensor& input,
    std::size_t position_offset,
    double base
) {
    if (input.rank() != 3) {
        throw std::invalid_argument(
            "RoPE input must have shape "
            "[sequence, heads, head dimension]"
        );
    }

    if (!std::isfinite(base) || base <= 0.0) {
        throw std::invalid_argument(
            "RoPE base must be finite and greater than zero"
        );
    }

    const std::size_t sequence_length =
        input.shape()[0];

    const std::size_t head_count =
        input.shape()[1];

    const std::size_t head_dimension =
        input.shape()[2];

    if (head_dimension % 2 != 0) {
        throw std::invalid_argument(
            "RoPE head dimension must be even"
        );
    }

    Tensor output(input.shape());

    const float* input_data = input.data();
    float* output_data = output.data();

    const std::size_t token_stride =
        head_count * head_dimension;

    for (
        std::size_t position = 0;
        position < sequence_length;
        ++position
    ) {
        const double absolute_position =
            static_cast<double>(position_offset) +
            static_cast<double>(position);

        for (
            std::size_t head = 0;
            head < head_count;
            ++head
        ) {
            const std::size_t head_offset =
                position * token_stride +
                head * head_dimension;

            for (
                std::size_t pair = 0;
                pair < head_dimension / 2;
                ++pair
            ) {
                const std::size_t first_index =
                    head_offset + pair * 2;

                const std::size_t second_index =
                    first_index + 1;

                const double frequency_exponent =
                    -2.0 *
                    static_cast<double>(pair) /
                    static_cast<double>(head_dimension);

                const double inverse_frequency =
                    std::pow(base, frequency_exponent);

                const double angle =
                    absolute_position * inverse_frequency;

                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);

                const double first_value =
                    static_cast<double>(
                        input_data[first_index]
                    );

                const double second_value =
                    static_cast<double>(
                        input_data[second_index]
                    );

                output_data[first_index] =
                    static_cast<float>(
                        first_value * cosine -
                        second_value * sine
                    );

                output_data[second_index] =
                    static_cast<float>(
                        first_value * sine +
                        second_value * cosine
                    );
            }
        }
    }

    return output;
}

}  // namespace edge_llm