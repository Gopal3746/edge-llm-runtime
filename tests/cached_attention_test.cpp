#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "edge_llm/attention.hpp"
#include "edge_llm/kv_cache.hpp"
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

edge_llm::CausalSelfAttention make_attention() {
    return edge_llm::CausalSelfAttention(
        identity_weight(2),
        identity_weight(2),
        identity_weight(2),
        identity_weight(2),
        1
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
    using edge_llm::KVCache;
    using edge_llm::Tensor;

    const auto attention = make_attention();

    const Tensor complete_sequence(
        {3, 2},
        {
            1.0F, 0.0F,
            0.0F, 1.0F,
            1.0F, 1.0F
        }
    );

    const Tensor prompt(
        {2, 2},
        {
            1.0F, 0.0F,
            0.0F, 1.0F
        }
    );

    const Tensor next_token(
        {1, 2},
        {1.0F, 1.0F}
    );

    const Tensor uncached_output =
        attention.forward(complete_sequence);

    KVCache cache(4, 1, 2);

    const Tensor prefill_output =
        attention.prefill(prompt, cache);

    expect(
        cache.size() == 2,
        "Prefill should cache every prompt token"
    );

    expect(
        prefill_output.shape() == Tensor::Shape({2, 2}),
        "Prefill should return one output per prompt token"
    );

    for (
        std::size_t index = 0;
        index < prefill_output.size();
        ++index
    ) {
        expect(
            nearly_equal(
                prefill_output.at(index),
                uncached_output.at(index)
            ),
            "Prefill output should match uncached attention"
        );
    }

    const Tensor decode_output =
        attention.decode(next_token, cache);

    expect(
        cache.size() == 3,
        "Decode should append one token to the cache"
    );

    expect(
        decode_output.shape() == Tensor::Shape({1, 2}),
        "Decode should return one output row"
    );

    expect(
        nearly_equal(
            decode_output.at(0),
            uncached_output.at(4)
        ) &&
        nearly_equal(
            decode_output.at(1),
            uncached_output.at(5)
        ),
        "Cached decode should match the final uncached output"
    );

    expect_throws<std::logic_error>(
        [&attention, &prompt, &cache] {
            static_cast<void>(
                attention.prefill(prompt, cache)
            );
        },
        "Prefill should reject an already populated cache"
    );

    expect_throws<std::logic_error>(
        [&attention, &next_token] {
            KVCache empty_cache(4, 1, 2);

            static_cast<void>(
                attention.decode(
                    next_token,
                    empty_cache
                )
            );
        },
        "Decode should reject an empty cache"
    );

    expect_throws<std::invalid_argument>(
        [&attention, &prompt] {
            KVCache other_cache(4, 1, 2);

            attention.prefill(prompt, other_cache);

            static_cast<void>(
                attention.decode(prompt, other_cache)
            );
        },
        "Decode should require exactly one token"
    );

    expect_throws<std::invalid_argument>(
        [&attention, &prompt] {
            KVCache wrong_cache(4, 2, 1);

            static_cast<void>(
                attention.prefill(
                    prompt,
                    wrong_cache
                )
            );
        },
        "Attention should reject incompatible cache dimensions"
    );

    expect_throws<std::length_error>(
        [&attention, &prompt, &next_token] {
            KVCache full_cache(2, 1, 2);

            attention.prefill(prompt, full_cache);

            static_cast<void>(
                attention.decode(
                    next_token,
                    full_cache
                )
            );
        },
        "Decode should reject a full cache"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " cached-attention test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "All cached-attention tests passed\n";

    return EXIT_SUCCESS;
}
