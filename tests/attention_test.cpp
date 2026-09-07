#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "edge_llm/attention.hpp"
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
    using edge_llm::Tensor;

    const CausalSelfAttention single_head_attention(
        identity_weight(2),
        identity_weight(2),
        identity_weight(2),
        identity_weight(2),
        1
    );

    expect(
        single_head_attention.model_dimension() == 2,
        "Attention should expose its model dimension"
    );

    expect(
        single_head_attention.head_count() == 1,
        "Attention should expose its head count"
    );

    expect(
        single_head_attention.head_dimension() == 2,
        "Attention should expose its head dimension"
    );

    const Tensor first_sequence(
        {2, 2},
        {
            1.0F, 0.0F,
            0.0F, 1.0F
        }
    );

    const Tensor second_sequence(
        {2, 2},
        {
            1.0F, 0.0F,
            10.0F, -5.0F
        }
    );

    const Tensor first_output =
        single_head_attention.forward(first_sequence);

    const Tensor second_output =
        single_head_attention.forward(second_sequence);

    expect(
        first_output.shape() == first_sequence.shape(),
        "Attention should preserve sequence and model dimensions"
    );

    expect(
        nearly_equal(first_output.at(0), 1.0F) &&
        nearly_equal(first_output.at(1), 0.0F),
        "The first token should attend only to itself"
    );

    expect(
        nearly_equal(first_output.at(0), second_output.at(0)) &&
        nearly_equal(first_output.at(1), second_output.at(1)),
        "Changing a future token must not affect an earlier token"
    );

    const CausalSelfAttention two_head_attention(
        identity_weight(4),
        identity_weight(4),
        identity_weight(4),
        identity_weight(4),
        2
    );

    const Tensor single_token(
        {1, 4},
        {1.0F, 2.0F, 3.0F, 4.0F}
    );

    const Tensor single_token_output =
        two_head_attention.forward(single_token, 7);

    expect_values(
        single_token_output,
        {1.0F, 2.0F, 3.0F, 4.0F},
        "A single token with identity projections should pass through"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const CausalSelfAttention invalid(
                identity_weight(2),
                identity_weight(2),
                identity_weight(2),
                identity_weight(2),
                0
            );
        },
        "Attention should reject zero heads"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const CausalSelfAttention invalid(
                identity_weight(4),
                identity_weight(4),
                identity_weight(4),
                identity_weight(4),
                3
            );
        },
        "Model dimension should be divisible by head count"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const CausalSelfAttention invalid(
                identity_weight(6),
                identity_weight(6),
                identity_weight(6),
                identity_weight(6),
                2
            );
        },
        "RoPE should require an even head dimension"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const CausalSelfAttention invalid(
                identity_weight(2),
                identity_weight(4),
                identity_weight(2),
                identity_weight(2),
                1
            );
        },
        "Attention should reject incompatible projections"
    );

    expect_throws<std::invalid_argument>(
        [&single_head_attention] {
            const Tensor rank_one_input(
                {2},
                {1.0F, 2.0F}
            );

            static_cast<void>(
                single_head_attention.forward(
                    rank_one_input
                )
            );
        },
        "Attention should require rank-2 input"
    );

    expect_throws<std::invalid_argument>(
        [&single_head_attention] {
            const Tensor wrong_width(
                {1, 4},
                {1.0F, 2.0F, 3.0F, 4.0F}
            );

            static_cast<void>(
                single_head_attention.forward(
                    wrong_width
                )
            );
        },
        "Attention should reject an incorrect input width"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " attention test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All attention tests passed\n";
    return EXIT_SUCCESS;
}