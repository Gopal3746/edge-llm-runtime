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

void expect_same_tensor(
    const edge_llm::Tensor& actual,
    const edge_llm::Tensor& expected,
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
                expected.at(index)
            )
        ) {
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

edge_llm::TransformerBlock make_block() {
    edge_llm::CausalSelfAttention attention(
        identity_weight(2),
        identity_weight(2),
        identity_weight(2),
        identity_weight(2),
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

edge_llm::TransformerModel make_model() {
    std::vector<edge_llm::TransformerBlock> blocks;

    blocks.push_back(make_block());
    blocks.push_back(make_block());

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
    using edge_llm::Tensor;
    using edge_llm::TransformerModel;

    const TransformerModel model = make_model();

    TransformerModel::LayerCaches caches =
        model.create_caches(6);

    expect(
        caches.size() == model.block_count(),
        "Model should create one cache per transformer block"
    );

    expect(
        caches[0].capacity() == 6 &&
        caches[1].capacity() == 6,
        "Every model cache should use the requested capacity"
    );

    const std::vector<std::size_t> prompt = {0, 1};

    const Tensor full_prompt_logits =
        model.forward(prompt);

    const Tensor cached_prompt_logits =
        model.prefill(prompt, caches);

    expect_same_tensor(
        cached_prompt_logits,
        full_prompt_logits,
        "Cached prefill logits should match full forward logits"
    );

    expect(
        caches[0].size() == prompt.size() &&
        caches[1].size() == prompt.size(),
        "Prefill should populate every layer cache"
    );

    constexpr std::size_t next_token = 2;

    const Tensor cached_decode_logits =
        model.decode(next_token, caches);

    const Tensor full_decode_logits =
        model.forward({0, 1, next_token});

    const std::size_t vocabulary_size =
        model.vocabulary_size();

    const std::size_t final_row_offset =
        (
            full_decode_logits.shape()[0] - 1
        ) * vocabulary_size;

    const Tensor expected_decode_logits(
        {1, vocabulary_size},
        {
            full_decode_logits.at(final_row_offset),
            full_decode_logits.at(final_row_offset + 1),
            full_decode_logits.at(final_row_offset + 2)
        }
    );

    expect_same_tensor(
        cached_decode_logits,
        expected_decode_logits,
        "Cached decode logits should match the final full-forward row"
    );

    expect(
        caches[0].size() == 3 &&
        caches[1].size() == 3,
        "Decode should append one token to every layer cache"
    );

    const std::vector<std::size_t> uncached_generation =
        model.generate({0, 1}, 3);

    const std::vector<std::size_t> cached_generation =
        model.generate_cached({0, 1}, 3);

    expect(
        cached_generation == uncached_generation,
        "Cached generation should match uncached greedy generation"
    );

    const std::vector<std::size_t> unchanged =
        model.generate_cached({0, 1}, 0);

    expect(
        unchanged ==
            std::vector<std::size_t>({0, 1}),
        "Cached generation of zero tokens should preserve the prompt"
    );

    expect_throws<std::invalid_argument>(
        [&model, &prompt] {
            TransformerModel::LayerCaches missing_caches;

            (void)model.prefill(
                prompt,
                missing_caches
            );
        },
        "Prefill should require one cache per block"
    );

    expect_throws<std::logic_error>(
        [&model, &prompt] {
            TransformerModel::LayerCaches local_caches =
                model.create_caches(4);

            (void)model.prefill(
                prompt,
                local_caches
            );

            (void)model.prefill(
                prompt,
                local_caches
            );
        },
        "Prefill should reject non-empty caches"
    );

    expect_throws<std::logic_error>(
        [&model] {
            TransformerModel::LayerCaches empty_caches =
                model.create_caches(4);

            (void)model.decode(
                0,
                empty_caches
            );
        },
        "Decode should require populated caches"
    );

    expect_throws<std::length_error>(
        [&model, &prompt] {
            TransformerModel::LayerCaches small_caches =
                model.create_caches(1);

            (void)model.prefill(
                prompt,
                small_caches
            );
        },
        "Prefill should reject prompts larger than cache capacity"
    );

    expect_throws<std::invalid_argument>(
        [&model] {
            (void)model.generate_cached({}, 1);
        },
        "Cached generation should reject an empty prompt"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " cached-model test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "All cached-model tests passed\n";

    return EXIT_SUCCESS;
}
