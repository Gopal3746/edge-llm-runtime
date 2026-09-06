#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "edge_llm/layers.hpp"
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
    using edge_llm::Embedding;
    using edge_llm::Linear;
    using edge_llm::Tensor;

    const Embedding embedding{
        Tensor(
            {3, 2},
            {
                0.1F, 0.2F,
                1.0F, 2.0F,
                3.0F, 4.0F
            }
        )
    };

    expect(
        embedding.vocabulary_size() == 3,
        "Embedding should expose its vocabulary size"
    );

    expect(
        embedding.embedding_dimension() == 2,
        "Embedding should expose its vector dimension"
    );

    const Tensor embedded =
        embedding.forward({2, 0, 1});

    expect(
        embedded.shape() == Tensor::Shape({3, 2}),
        "Embedding output should have one row per token"
    );

    expect_values(
        embedded,
        {
            3.0F, 4.0F,
            0.1F, 0.2F,
            1.0F, 2.0F
        },
        "Embedding should copy the selected vocabulary rows"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const Embedding invalid{
                Tensor(
                    {4},
                    {1.0F, 2.0F, 3.0F, 4.0F}
                )
            };
        },
        "Embedding should require rank-2 weights"
    );

    expect_throws<std::invalid_argument>(
        [&embedding] {
            static_cast<void>(
                embedding.forward(
                    std::vector<std::size_t>{}
                )
            );
        },
        "Embedding should reject an empty token sequence"
    );

    expect_throws<std::out_of_range>(
        [&embedding] {
            static_cast<void>(
                embedding.forward({0, 3})
            );
        },
        "Embedding should reject an unknown token ID"
    );

    const Linear linear{
        Tensor(
            {2, 3},
            {
                1.0F, 0.0F, -1.0F,
                0.5F, 0.5F, 0.5F
            }
        )
    };

    expect(
        linear.input_features() == 3,
        "Linear should expose its input feature count"
    );

    expect(
        linear.output_features() == 2,
        "Linear should expose its output feature count"
    );

    const Tensor linear_input(
        {2, 3},
        {
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F
        }
    );

    const Tensor linear_output =
        linear.forward(linear_input);

    expect(
        linear_output.shape() == Tensor::Shape({2, 2}),
        "Linear output should have one row per input row"
    );

    expect_values(
        linear_output,
        {
            -2.0F, 3.0F,
            -2.0F, 7.5F
        },
        "Linear should calculate X multiplied by weight transpose"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const Linear invalid{
                Tensor(
                    {3},
                    {1.0F, 2.0F, 3.0F}
                )
            };
        },
        "Linear should require rank-2 weights"
    );

    expect_throws<std::invalid_argument>(
        [&linear] {
            const Tensor rank_one_input(
                {3},
                {1.0F, 2.0F, 3.0F}
            );

            static_cast<void>(
                linear.forward(rank_one_input)
            );
        },
        "Linear should require rank-2 input"
    );

    expect_throws<std::invalid_argument>(
        [&linear] {
            const Tensor wrong_width_input(
                {2, 2},
                {
                    1.0F, 2.0F,
                    3.0F, 4.0F
                }
            );

            static_cast<void>(
                linear.forward(wrong_width_input)
            );
        },
        "Linear should reject an incompatible input width"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " layer test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All layer tests passed\n";
    return EXIT_SUCCESS;
}