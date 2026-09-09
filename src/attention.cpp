#include "edge_llm/attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "edge_llm/nn.hpp"
#include "edge_llm/rope.hpp"
#include "edge_llm/tensor.hpp"

namespace edge_llm {
namespace {

Tensor split_heads(
    const Tensor& projection,
    std::size_t head_count
) {
    const std::size_t sequence_length =
        projection.shape()[0];

    const std::size_t model_dimension =
        projection.shape()[1];

    const std::size_t head_dimension =
        model_dimension / head_count;

    Tensor result({
        sequence_length,
        head_count,
        head_dimension
    });

    std::copy_n(
        projection.data(),
        projection.size(),
        result.data()
    );

    return result;
}

Tensor merge_heads(
    const Tensor& input
) {
    const std::size_t sequence_length =
        input.shape()[0];

    const std::size_t head_count =
        input.shape()[1];

    const std::size_t head_dimension =
        input.shape()[2];

    Tensor result({
        sequence_length,
        head_count * head_dimension
    });

    std::copy_n(
        input.data(),
        input.size(),
        result.data()
    );

    return result;
}

Tensor attend(
    const Tensor& queries,
    const float* key_data,
    const float* value_data,
    std::size_t key_count,
    std::size_t query_position_offset,
    std::size_t key_position_offset
) {
    const std::size_t query_count =
        queries.shape()[0];

    const std::size_t head_count =
        queries.shape()[1];

    const std::size_t head_dimension =
        queries.shape()[2];

    Tensor scores({
        head_count,
        query_count,
        key_count
    });

    const float* query_data = queries.data();
    float* score_data = scores.data();

    const float negative_infinity =
        -std::numeric_limits<float>::infinity();

    const double scale =
        1.0 /
        std::sqrt(
            static_cast<double>(head_dimension)
        );

    for (
        std::size_t head = 0;
        head < head_count;
        ++head
    ) {
        for (
            std::size_t query_position = 0;
            query_position < query_count;
            ++query_position
        ) {
            const std::size_t absolute_query_position =
                query_position_offset + query_position;

            for (
                std::size_t key_position = 0;
                key_position < key_count;
                ++key_position
            ) {
                const std::size_t absolute_key_position =
                    key_position_offset + key_position;

                const std::size_t score_index =
                    (
                        head * query_count +
                        query_position
                    ) * key_count +
                    key_position;

                if (
                    absolute_key_position >
                    absolute_query_position
                ) {
                    score_data[score_index] =
                        negative_infinity;

                    continue;
                }

                const std::size_t query_offset =
                    (
                        query_position * head_count +
                        head
                    ) * head_dimension;

                const std::size_t key_offset =
                    (
                        key_position * head_count +
                        head
                    ) * head_dimension;

                double dot_product = 0.0;

                for (
                    std::size_t dimension = 0;
                    dimension < head_dimension;
                    ++dimension
                ) {
                    dot_product +=
                        static_cast<double>(
                            query_data[
                                query_offset + dimension
                            ]
                        ) *
                        static_cast<double>(
                            key_data[
                                key_offset + dimension
                            ]
                        );
                }

                score_data[score_index] =
                    static_cast<float>(
                        dot_product * scale
                    );
            }
        }
    }

    const Tensor probabilities =
        softmax_last_dim(scores);

    Tensor context({
        query_count,
        head_count,
        head_dimension
    });

    const float* probability_data =
        probabilities.data();

    float* context_data = context.data();

    for (
        std::size_t query_position = 0;
        query_position < query_count;
        ++query_position
    ) {
        for (
            std::size_t head = 0;
            head < head_count;
            ++head
        ) {
            for (
                std::size_t dimension = 0;
                dimension < head_dimension;
                ++dimension
            ) {
                double weighted_sum = 0.0;

                for (
                    std::size_t key_position = 0;
                    key_position < key_count;
                    ++key_position
                ) {
                    const std::size_t probability_index =
                        (
                            head * query_count +
                            query_position
                        ) * key_count +
                        key_position;

                    const std::size_t value_offset =
                        (
                            key_position * head_count +
                            head
                        ) * head_dimension;

                    weighted_sum +=
                        static_cast<double>(
                            probability_data[
                                probability_index
                            ]
                        ) *
                        static_cast<double>(
                            value_data[
                                value_offset + dimension
                            ]
                        );
                }

                const std::size_t context_offset =
                    (
                        query_position * head_count +
                        head
                    ) * head_dimension;

                context_data[
                    context_offset + dimension
                ] = static_cast<float>(weighted_sum);
            }
        }
    }

    return context;
}

}  // namespace

CausalSelfAttention::CausalSelfAttention(
    Tensor query_weight,
    Tensor key_weight,
    Tensor value_weight,
    Tensor output_weight,
    std::size_t head_count,
    double rope_base
)
    : query_projection_(std::move(query_weight)),
      key_projection_(std::move(key_weight)),
      value_projection_(std::move(value_weight)),
      output_projection_(std::move(output_weight)),
      head_count_(head_count),
      rope_base_(rope_base) {
    if (head_count_ == 0) {
        throw std::invalid_argument(
            "Attention head count must be greater than zero"
        );
    }

    if (!std::isfinite(rope_base_) || rope_base_ <= 0.0) {
        throw std::invalid_argument(
            "Attention RoPE base must be finite and positive"
        );
    }

    const std::size_t dimension =
        query_projection_.input_features();

    if (
        query_projection_.output_features() != dimension ||
        key_projection_.input_features() != dimension ||
        key_projection_.output_features() != dimension ||
        value_projection_.input_features() != dimension ||
        value_projection_.output_features() != dimension ||
        output_projection_.input_features() != dimension ||
        output_projection_.output_features() != dimension
    ) {
        throw std::invalid_argument(
            "All attention projections must have shape "
            "[model dimension, model dimension]"
        );
    }

    if (dimension % head_count_ != 0) {
        throw std::invalid_argument(
            "Model dimension must be divisible by head count"
        );
    }

    if (head_dimension() % 2 != 0) {
        throw std::invalid_argument(
            "Attention head dimension must be even for RoPE"
        );
    }
}

Tensor CausalSelfAttention::forward(
    const Tensor& input,
    std::size_t position_offset
) const {
    validate_input(input);

    const Tensor query_heads =
        split_heads(
            query_projection_.forward(input),
            head_count_
        );

    const Tensor key_heads =
        split_heads(
            key_projection_.forward(input),
            head_count_
        );

    const Tensor value_heads =
        split_heads(
            value_projection_.forward(input),
            head_count_
        );

    const Tensor rotated_queries =
        apply_rope(
            query_heads,
            position_offset,
            rope_base_
        );

    const Tensor rotated_keys =
        apply_rope(
            key_heads,
            position_offset,
            rope_base_
        );

    const Tensor context =
        attend(
            rotated_queries,
            rotated_keys.data(),
            value_heads.data(),
            input.shape()[0],
            position_offset,
            position_offset
        );

    return output_projection_.forward(
        merge_heads(context)
    );
}

Tensor CausalSelfAttention::prefill(
    const Tensor& input,
    KVCache& cache
) const {
    validate_input(input);
    validate_cache(cache);

    if (!cache.empty()) {
        throw std::logic_error(
            "Attention prefill requires an empty cache"
        );
    }

    if (input.shape()[0] > cache.capacity()) {
        throw std::length_error(
            "Prompt exceeds KV cache capacity"
        );
    }

    const Tensor query_heads =
        split_heads(
            query_projection_.forward(input),
            head_count_
        );

    const Tensor key_heads =
        split_heads(
            key_projection_.forward(input),
            head_count_
        );

    const Tensor value_heads =
        split_heads(
            value_projection_.forward(input),
            head_count_
        );

    const Tensor rotated_queries =
        apply_rope(
            query_heads,
            0,
            rope_base_
        );

    const Tensor rotated_keys =
        apply_rope(
            key_heads,
            0,
            rope_base_
        );

    const Tensor context =
        attend(
            rotated_queries,
            rotated_keys.data(),
            value_heads.data(),
            input.shape()[0],
            0,
            0
        );

    const Tensor output =
        output_projection_.forward(
            merge_heads(context)
        );

    cache.append(
        rotated_keys,
        value_heads
    );

    return output;
}

Tensor CausalSelfAttention::decode(
    const Tensor& input,
    KVCache& cache
) const {
    validate_input(input);
    validate_cache(cache);

    if (input.shape()[0] != 1) {
        throw std::invalid_argument(
            "Attention decode requires exactly one token"
        );
    }

    if (cache.empty()) {
        throw std::logic_error(
            "Attention decode requires a populated cache"
        );
    }

    if (cache.full()) {
        throw std::length_error(
            "KV cache has no room for another token"
        );
    }

    const std::size_t token_position =
        cache.size();

    const Tensor query_heads =
        split_heads(
            query_projection_.forward(input),
            head_count_
        );

    const Tensor key_heads =
        split_heads(
            key_projection_.forward(input),
            head_count_
        );

    const Tensor value_heads =
        split_heads(
            value_projection_.forward(input),
            head_count_
        );

    const Tensor rotated_queries =
        apply_rope(
            query_heads,
            token_position,
            rope_base_
        );

    const Tensor rotated_keys =
        apply_rope(
            key_heads,
            token_position,
            rope_base_
        );

    cache.append(
        rotated_keys,
        value_heads
    );

    const Tensor context =
        attend(
            rotated_queries,
            cache.key_data(),
            cache.value_data(),
            cache.size(),
            token_position,
            0
        );

    return output_projection_.forward(
        merge_heads(context)
    );
}

void CausalSelfAttention::validate_input(
    const Tensor& input
) const {
    if (
        input.rank() != 2 ||
        input.shape()[1] != model_dimension()
    ) {
        throw std::invalid_argument(
            "Attention input must have shape "
            "[sequence, model dimension]"
        );
    }
}

void CausalSelfAttention::validate_cache(
    const KVCache& cache
) const {
    if (
        cache.head_count() != head_count_ ||
        cache.head_dimension() != head_dimension()
    ) {
        throw std::invalid_argument(
            "KV cache dimensions do not match attention"
        );
    }
}

std::size_t
CausalSelfAttention::model_dimension() const noexcept {
    return query_projection_.input_features();
}

std::size_t
CausalSelfAttention::head_count() const noexcept {
    return head_count_;
}

std::size_t
CausalSelfAttention::head_dimension() const noexcept {
    return model_dimension() / head_count_;
}

}  // namespace edge_llm
