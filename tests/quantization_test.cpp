#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "edge_llm/layers.hpp"
#include "edge_llm/quantization.hpp"
#include "edge_llm/tensor.hpp"

namespace {

int failure_count = 0;

void expect(
    bool condition,
    std::string_view message
) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failure_count;
    }
}

bool nearly_equal(
    float left,
    float right,
    float tolerance = 2.0e-2F
) {
    return std::fabs(left - right) <= tolerance;
}

void expect_near_tensor(
    const edge_llm::Tensor& actual,
    const edge_llm::Tensor& expected,
    float tolerance,
    std::string_view message
) {
    if (actual.shape() != expected.shape()) {
        expect(false, message);
        return;
    }

    for (
        std::size_t index = 0;
        index < actual.size();
        ++index
    ) {
        if (
            !nearly_equal(
                actual.at(index),
                expected.at(index),
                tolerance
            )
        ) {
            expect(false, message);
            return;
        }
    }

    expect(true, message);
}

template <typename Exception, typename Function>
void expect_throws(
    Function function,
    std::string_view message
) {
    bool threw_expected_exception = false;

    try {
        function();
    } catch (const Exception&) {
        threw_expected_exception = true;
    } catch (...) {
        // A different exception does not satisfy the test.
    }

    expect(threw_expected_exception, message);
}

}  // namespace

int main() {
    using edge_llm::Linear;
    using edge_llm::QuantizedLinear;
    using edge_llm::Tensor;

    const Tensor float_weight(
        {3, 4},
        {
            1.0F, -0.5F, 0.25F, -1.5F,
            0.0F, 0.0F, 0.0F, 0.0F,
            0.1F, 0.2F, 0.3F, 0.4F
        }
    );

    const Linear float_linear(float_weight);
    const QuantizedLinear quantized_linear(
        float_weight
    );

    expect(
        quantized_linear.input_features() == 4,
        "Quantized linear should expose input features"
    );

    expect(
        quantized_linear.output_features() == 3,
        "Quantized linear should expose output features"
    );

    expect(
        quantized_linear.scale_at(0) > 0.0F,
        "Each quantized row should have a positive scale"
    );

    expect(
        quantized_linear.scale_at(1) == 1.0F,
        "An all-zero row should use a safe unit scale"
    );

    expect(
        quantized_linear.quantized_weight_at(0, 3) == -127,
        "The largest negative weight should map to -127"
    );

    expect(
        quantized_linear.quantized_weight_at(2, 3) == 127,
        "The largest positive weight should map to 127"
    );

    expect(
        quantized_linear.quantized_weight_bytes() <
            quantized_linear.float_weight_bytes(),
        "INT8 weights should use less storage than FP32 weights"
    );

    expect(
        quantized_linear.compression_ratio() > 1.0,
        "Quantized storage should provide compression"
    );

    const Tensor input(
        {2, 4},
        {
            1.0F, 2.0F, -1.0F, 0.5F,
            -2.0F, 1.0F, 0.25F, 3.0F
        }
    );

    const Tensor float_output =
        float_linear.forward(input);

    const Tensor quantized_output =
        quantized_linear.forward(input);

    expect_near_tensor(
        quantized_output,
        float_output,
        2.0e-2F,
        "INT8 linear output should remain close to FP32 output"
    );

    const Tensor dequantized_weight =
        quantized_linear.dequantized_weight();

    expect_near_tensor(
        dequantized_weight,
        float_weight,
        1.2e-2F,
        "Dequantized weights should remain close to FP32 weights"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const Tensor rank_one_weight(
                {4},
                {
                    1.0F,
                    2.0F,
                    3.0F,
                    4.0F
                }
            );

            (void)QuantizedLinear(
                rank_one_weight
            );
        },
        "Quantization should reject non-matrix weights"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const Tensor non_finite_weight(
                {1, 2},
                {
                    1.0F,
                    std::numeric_limits<float>::infinity()
                }
            );

            (void)QuantizedLinear(
                non_finite_weight
            );
        },
        "Quantization should reject non-finite weights"
    );

    expect_throws<std::invalid_argument>(
        [&quantized_linear] {
            const Tensor invalid_input(
                {1, 3},
                {
                    1.0F,
                    2.0F,
                    3.0F
                }
            );

            (void)quantized_linear.forward(
                invalid_input
            );
        },
        "Quantized linear should reject incompatible input shapes"
    );

    expect_throws<std::out_of_range>(
        [&quantized_linear] {
            (void)quantized_linear.scale_at(3);
        },
        "Scale lookup should validate the output row"
    );

    expect_throws<std::out_of_range>(
        [&quantized_linear] {
            (void)quantized_linear.quantized_weight_at(
                0,
                4
            );
        },
        "Quantized weight lookup should validate indices"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " quantization test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "All quantization tests passed\n";

    return EXIT_SUCCESS;
}
