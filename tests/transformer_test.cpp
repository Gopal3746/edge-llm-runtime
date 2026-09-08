#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "edge_llm/attention.hpp"
#include "edge_llm/feed_forward.hpp"
#include "edge_llm/tensor.hpp"
#include "edge_llm/transformer.hpp"

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
    float tolerance = 1.0e-5F
) {
    return std::fabs(left - right) <= tolerance;
}

void expect_values(
    const edge_llm::Tensor& tensor,
    const std::vector<float>& expected,
    std::string_view message
) {
    if (tensor.size() != expected.size()) {
        expect(false, message);
        return;
    }

    for (
        std::size_t index = 0;
        index < expected.size();
        ++index
    ) {
        if (!nearly_equal(tensor.at(index), expected[index])) {
            expect(false, message);
            return;
        }
    }

    expect(true, message);
}

edge_llm::Tensor identity_weight(
    std::size_t dimension
) {
    std::vector<float> values(
        dimension * dimension,
        0.0F
    );

    for (
        std::size_t index = 0;
        index < dimension;
        ++index
    ) {
        values[index * dimension + index] = 1.0F;
    }

    return edge_llm::Tensor(
        {dimension, dimension},
        std::move(values)
    );
}

edge_llm::Tensor zero_weight(
    std::size_t output_dimension,
    std::size_t input_dimension
) {
    return edge_llm::Tensor({
        output_dimension,
        input_dimension
    });
}

edge_llm::Tensor ones(
    std::size_t size
) {
    return edge_llm::Tensor(
        {size},
        std::vector<float>(size, 1.0F)
    );
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
    using edge_llm::CausalSelfAttention;
    using edge_llm::SwiGLUFeedForward;
    using edge_llm::Tensor;
    using edge_llm::TransformerBlock;

    const SwiGLUFeedForward identity_feed_forward(
        identity_weight(2),
        identity_weight(2),
        identity_weight(2)
    );

    const Tensor feed_forward_input(
        {1, 2},
        {1.0F, 2.0F}
    );

    const Tensor feed_forward_output =
        identity_feed_forward.forward(
            feed_forward_input
        );

    const float sigmoid_one =
        1.0F /
        (
            1.0F +
            static_cast<float>(std::exp(-1.0))
        );

    const float sigmoid_two =
        1.0F /
        (
            1.0F +
            static_cast<float>(std::exp(-2.0))
        );

    expect_values(
        feed_forward_output,
        {
            sigmoid_one,
            4.0F * sigmoid_two
        },
        "SwiGLU should apply SiLU gating element-wise"
    );

    expect(
        identity_feed_forward.model_dimension() == 2,
        "SwiGLU should expose its model dimension"
    );

    expect(
        identity_feed_forward.hidden_dimension() == 2,
        "SwiGLU should expose its hidden dimension"
    );

    const CausalSelfAttention zero_attention(
        zero_weight(2, 2),
        zero_weight(2, 2),
        zero_weight(2, 2),
        zero_weight(2, 2),
        1
    );

    const SwiGLUFeedForward zero_feed_forward(
        zero_weight(2, 2),
        zero_weight(2, 2),
        zero_weight(2, 2)
    );

    const TransformerBlock block(
        zero_attention,
        zero_feed_forward,
        ones(2),
        ones(2)
    );

    const Tensor block_input(
        {2, 2},
        {
            1.0F, -2.0F,
            3.0F, 4.0F
        }
    );

    const Tensor block_output =
        block.forward(block_input, 5);

    expect(
        block_output.shape() == block_input.shape(),
        "Transformer block should preserve input shape"
    );

    expect_values(
        block_output,
        {
            1.0F, -2.0F,
            3.0F, 4.0F
        },
        "Zero sublayers should preserve input through residuals"
    );

    expect(
        block.model_dimension() == 2,
        "Transformer block should expose model dimension"
    );

    expect(
        block.hidden_dimension() == 2,
        "Transformer block should expose hidden dimension"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const SwiGLUFeedForward invalid(
                zero_weight(3, 2),
                zero_weight(4, 2),
                zero_weight(2, 3)
            );
        },
        "SwiGLU should reject incompatible projections"
    );

    expect_throws<std::invalid_argument>(
        [&zero_attention, &zero_feed_forward] {
            const TransformerBlock invalid(
                zero_attention,
                zero_feed_forward,
                ones(3),
                ones(2)
            );
        },
        "Transformer block should reject an incorrect norm size"
    );

    expect_throws<std::invalid_argument>(
        [&zero_attention, &zero_feed_forward] {
            const TransformerBlock invalid(
                zero_attention,
                zero_feed_forward,
                ones(2),
                ones(2),
                0.0F
            );
        },
        "Transformer block should reject non-positive epsilon"
    );

    expect_throws<std::invalid_argument>(
        [&block] {
            const Tensor wrong_width(
                {1, 4},
                {1.0F, 2.0F, 3.0F, 4.0F}
            );

            static_cast<void>(
                block.forward(wrong_width)
            );
        },
        "Transformer block should reject incorrect input width"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " transformer test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All transformer tests passed\n";
    return EXIT_SUCCESS;
}