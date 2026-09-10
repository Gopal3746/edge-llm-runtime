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
#include "edge_llm/kv_cache.hpp"
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
        if (
            !nearly_equal(
                tensor.at(index),
                expected[index]
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

}  // namespace

int main() {
    using edge_llm::KVCache;
    using edge_llm::Tensor;
    using edge_llm::TransformerBlock;

    const TransformerBlock block = make_block();

    const Tensor complete_input(
        {3, 2},
        {
            1.0F, 2.0F,
            3.0F, 1.0F,
            2.0F, 4.0F
        }
    );

    const Tensor prompt(
        {2, 2},
        {
            1.0F, 2.0F,
            3.0F, 1.0F
        }
    );

    const Tensor next_token(
        {1, 2},
        {
            2.0F, 4.0F
        }
    );

    const Tensor full_output =
        block.forward(complete_input);

    KVCache cache(
        4,
        block.head_count(),
        block.head_dimension()
    );

    const Tensor prefill_output =
        block.prefill(prompt, cache);

    expect(
        prefill_output.shape() == Tensor::Shape({2, 2}),
        "Prefill should preserve the prompt shape"
    );

    expect_values(
        prefill_output,
        {
            full_output.at(0),
            full_output.at(1),
            full_output.at(2),
            full_output.at(3)
        },
        "Cached block prefill should match full forward output"
    );

    expect(
        cache.size() == 2,
        "Prefill should cache both prompt tokens"
    );

    const Tensor decode_output =
        block.decode(next_token, cache);

    expect(
        decode_output.shape() == Tensor::Shape({1, 2}),
        "Decode should return one token representation"
    );

    expect_values(
        decode_output,
        {
            full_output.at(4),
            full_output.at(5)
        },
        "Cached block decode should match the final full-forward row"
    );

    expect(
        cache.size() == 3,
        "Decode should append one token to the cache"
    );

    expect(
        block.head_count() == 1,
        "Transformer block should expose attention head count"
    );

    expect(
        block.head_dimension() == 2,
        "Transformer block should expose attention head dimension"
    );

    expect_throws<std::invalid_argument>(
        [&block, &prompt, &cache] {
            (void)block.decode(prompt, cache);
        },
        "Decode should reject more than one input token"
    );

    expect_throws<std::logic_error>(
        [&block, &next_token] {
            KVCache empty_cache(
                4,
                block.head_count(),
                block.head_dimension()
            );

            (void)block.decode(
                next_token,
                empty_cache
            );
        },
        "Decode should reject an empty cache"
    );

    expect_throws<std::logic_error>(
        [&block, &prompt, &cache] {
            (void)block.prefill(prompt, cache);
        },
        "Prefill should reject a non-empty cache"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " cached-transformer test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "All cached-transformer tests passed\n";

    return EXIT_SUCCESS;
}
