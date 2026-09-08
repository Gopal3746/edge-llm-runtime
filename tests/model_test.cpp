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
#include "edge_llm/layers.hpp"
#include "edge_llm/model.hpp"
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

edge_llm::Embedding make_embedding() {
    return edge_llm::Embedding{
        edge_llm::Tensor(
            {3, 2},
            {
                1.0F, 0.0F,
                0.0F, 1.0F,
                1.0F, 1.0F
            }
        )
    };
}

edge_llm::Linear make_output_projection() {
    return edge_llm::Linear{
        edge_llm::Tensor(
            {3, 2},
            {
                2.0F, 0.0F,
                0.0F, 2.0F,
                1.0F, 1.5F
            }
        )
    };
}

edge_llm::TransformerBlock make_zero_block() {
    edge_llm::CausalSelfAttention attention(
        zero_weight(2, 2),
        zero_weight(2, 2),
        zero_weight(2, 2),
        zero_weight(2, 2),
        1
    );

    edge_llm::SwiGLUFeedForward feed_forward(
        zero_weight(2, 2),
        zero_weight(2, 2),
        zero_weight(2, 2)
    );

    return edge_llm::TransformerBlock(
        std::move(attention),
        std::move(feed_forward),
        ones(2),
        ones(2)
    );
}

edge_llm::TransformerModel make_tiny_model() {
    std::vector<edge_llm::TransformerBlock> blocks;

    blocks.push_back(
        make_zero_block()
    );

    return edge_llm::TransformerModel(
        make_embedding(),
        std::move(blocks),
        ones(2),
        make_output_projection()
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
    using edge_llm::Linear;
    using edge_llm::Tensor;
    using edge_llm::TransformerBlock;
    using edge_llm::TransformerModel;

    const TransformerModel model =
        make_tiny_model();

    expect(
        model.vocabulary_size() == 3,
        "Model should expose vocabulary size"
    );

    expect(
        model.model_dimension() == 2,
        "Model should expose embedding dimension"
    );

    expect(
        model.block_count() == 1,
        "Model should expose transformer block count"
    );

    const Tensor logits =
        model.forward({0, 1});

    expect(
        logits.shape() == Tensor::Shape({2, 3}),
        "Model should produce one vocabulary row per token"
    );

    constexpr float epsilon = 1.0e-5F;

    const float normalized_value =
        1.0F / std::sqrt(0.5F + epsilon);

    expect(
        nearly_equal(logits.at(3), 0.0F),
        "Token one should produce zero for vocabulary item zero"
    );

    expect(
        nearly_equal(
            logits.at(4),
            2.0F * normalized_value
        ),
        "Token one should produce the expected highest logit"
    );

    expect(
        nearly_equal(
            logits.at(5),
            1.5F * normalized_value
        ),
        "Token one should produce the expected third logit"
    );

    const std::vector<std::size_t> generated =
        model.generate({1}, 3);

    expect(
        generated ==
            std::vector<std::size_t>({1, 1, 1, 1}),
        "Greedy generation should repeatedly select token one"
    );

    const std::vector<std::size_t> unchanged =
        model.generate({0, 2}, 0);

    expect(
        unchanged ==
            std::vector<std::size_t>({0, 2}),
        "Generating zero tokens should preserve the prompt"
    );

    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                TransformerModel(
                    make_embedding(),
                    std::vector<TransformerBlock>{},
                    ones(2),
                    make_output_projection()
                )
            );
        },
        "Model should require at least one transformer block"
    );

    expect_throws<std::invalid_argument>(
        [] {
            std::vector<TransformerBlock> blocks;
            blocks.push_back(make_zero_block());

            static_cast<void>(
                TransformerModel(
                    make_embedding(),
                    std::move(blocks),
                    ones(2),
                    Linear{
                        zero_weight(4, 2)
                    }
                )
            );
        },
        "Output vocabulary should match embedding vocabulary"
    );

    expect_throws<std::invalid_argument>(
        [&model] {
            static_cast<void>(
                model.generate({}, 1)
            );
        },
        "Generation should reject an empty prompt"
    );

    expect_throws<std::out_of_range>(
        [&model] {
            static_cast<void>(
                model.forward({3})
            );
        },
        "Model should reject token IDs outside the vocabulary"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " model test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All model tests passed\n";
    return EXIT_SUCCESS;
}