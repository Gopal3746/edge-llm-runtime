#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

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

    KVCache cache(3, 2, 2);

    expect(
        cache.capacity() == 3,
        "Cache should expose its token capacity"
    );

    expect(
        cache.head_count() == 2,
        "Cache should expose its head count"
    );

    expect(
        cache.head_dimension() == 2,
        "Cache should expose its head dimension"
    );

    expect(
        cache.empty() &&
        cache.size() == 0 &&
        cache.remaining_capacity() == 3,
        "A new cache should be empty"
    );

    const float* original_key_address =
        cache.key_data();

    const float* original_value_address =
        cache.value_data();

    const Tensor first_keys(
        {2, 2, 2},
        {
            0.0F, 1.0F,
            2.0F, 3.0F,
            4.0F, 5.0F,
            6.0F, 7.0F
        }
    );

    const Tensor first_values(
        {2, 2, 2},
        {
            10.0F, 11.0F,
            12.0F, 13.0F,
            14.0F, 15.0F,
            16.0F, 17.0F
        }
    );

    cache.append(first_keys, first_values);

    expect(
        cache.size() == 2 &&
        cache.remaining_capacity() == 1,
        "Append should advance the populated size"
    );

    expect(
        cache.key_at(1, 1, 0) == 6.0F,
        "Cache should preserve appended keys"
    );

    expect(
        cache.value_at(1, 0, 1) == 15.0F,
        "Cache should preserve appended values"
    );

    const Tensor second_keys(
        {1, 2, 2},
        {
            8.0F, 9.0F,
            10.0F, 11.0F
        }
    );

    const Tensor second_values(
        {1, 2, 2},
        {
            18.0F, 19.0F,
            20.0F, 21.0F
        }
    );

    cache.append(second_keys, second_values);

    expect(
        cache.full() &&
        cache.size() == 3 &&
        cache.remaining_capacity() == 0,
        "Cache should report when it reaches capacity"
    );

    expect(
        cache.key_at(2, 0, 0) == 8.0F &&
        cache.value_at(2, 1, 1) == 21.0F,
        "A second append should follow existing entries"
    );

    expect_throws<std::length_error>(
        [&cache, &second_keys, &second_values] {
            cache.append(
                second_keys,
                second_values
            );
        },
        "Cache should reject writes beyond capacity"
    );

    cache.clear();

    expect(
        cache.empty() &&
        cache.size() == 0 &&
        cache.remaining_capacity() == 3,
        "Clear should reset logical cache state"
    );

    expect(
        cache.key_data() == original_key_address &&
        cache.value_data() == original_value_address,
        "Clear should not reallocate cache storage"
    );

    cache.append(second_keys, second_values);

    expect(
        cache.key_at(0, 0, 0) == 8.0F &&
        cache.value_at(0, 1, 1) == 21.0F,
        "Append after clear should overwrite from position zero"
    );

    expect_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.key_at(1, 0, 0)
            );
        },
        "Cache should reject unpopulated positions"
    );

    expect_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.value_at(0, 2, 0)
            );
        },
        "Cache should reject invalid heads"
    );

    expect_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                KVCache(0, 2, 2)
            );
        },
        "Cache should reject zero capacity"
    );

    expect_throws<std::invalid_argument>(
        [] {
            KVCache other_cache(4, 2, 2);

            const Tensor rank_two_keys({1, 4});
            const Tensor rank_two_values({1, 4});

            other_cache.append(
                rank_two_keys,
                rank_two_values
            );
        },
        "Cache should require rank-3 tensors"
    );

    expect_throws<std::invalid_argument>(
        [] {
            KVCache other_cache(4, 2, 2);

            const Tensor wrong_heads(
                {1, 1, 2},
                {1.0F, 2.0F}
            );

            other_cache.append(
                wrong_heads,
                wrong_heads
            );
        },
        "Cache should reject incompatible head counts"
    );

    expect_throws<std::invalid_argument>(
        [] {
            KVCache other_cache(4, 2, 2);

            const Tensor keys({1, 2, 2});
            const Tensor values({1, 2, 4});

            other_cache.append(keys, values);
        },
        "Cache should require matching key and value shapes"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " KV-cache test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All KV-cache tests passed\n";
    return EXIT_SUCCESS;
}
