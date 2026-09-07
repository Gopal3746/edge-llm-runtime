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
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "Attention input must have shape "
            "[sequence, model dimension]"
        );
    }

    if (input.shape()[1] != model_dimension()) {
        throw std::invalid_argument(
            "Attention input width must match model dimension"
        );
    }

    const std::size_t sequence_length =
        input.shape()[0];

    const Tensor query_projection =
        query_projection_.forward(input);

    const Tensor key_projection =
        key_projection_.forward(input);

    const Tensor value_projection =
        value_projection_.forward(input);

    const Tensor query_heads =
        split_heads(query_projection, head_count_);

    const Tensor key_heads =
        split_heads(key_projection, head_count_);

    const Tensor value_heads =
        split_heads(value_projection, head_count_);

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

    Tensor scores({
        head_count_,
        sequence_length,
        sequence_length
    });

    float* score_data = scores.data();
    const float* query_data = rotated_queries.data();
    const float* key_data = rotated_keys.data();

    const float negative_infinity =
        -std::numeric_limits<float>::infinity();

    const double scale =
        1.0 /
        std::sqrt(
            static_cast<double>(head_dimension())
        );

    for (
        std::size_t head = 0;
        head < head_count_;
        ++head
    ) {
        for (
            std::size_t query_position = 0;
            query_position < sequence_length;
            ++query_position
        ) {
            for (
                std::size_t key_position = 0;
                key_position < sequence_length;
                ++key_position
            ) {
                const std::size_t score_index =
                    (
                        head * sequence_length +
                        query_position
                    ) * sequence_length +
                    key_position;

                if (key_position > query_position) {
                    score_data[score_index] =
                        negative_infinity;

                    continue;
                }

                const std::size_t query_offset =
                    (
                        query_position * head_count_ +
                        head
                    ) * head_dimension();

                const std::size_t key_offset =
                    (
                        key_position * head_count_ +
                        head
                    ) * head_dimension();

                double dot_product = 0.0;

                for (
                    std::size_t dimension = 0;
                    dimension < head_dimension();
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
        sequence_length,
        head_count_,
        head_dimension()
    });

    const float* probability_data =
        probabilities.data();

    const float* value_data =
        value_heads.data();

    float* context_data =
        context.data();

    for (
        std::size_t query_position = 0;
        query_position < sequence_length;
        ++query_position
    ) {
        for (
            std::size_t head = 0;
            head < head_count_;
            ++head
        ) {
            for (
                std::size_t dimension = 0;
                dimension < head_dimension();
                ++dimension
            ) {
                double weighted_sum = 0.0;

                for (
                    std::size_t key_position = 0;
                    key_position < sequence_length;
                    ++key_position
                ) {
                    const std::size_t probability_index =
                        (
                            head * sequence_length +
                            query_position
                        ) * sequence_length +
                        key_position;

                    const std::size_t value_offset =
                        (
                            key_position * head_count_ +
                            head
                        ) * head_dimension();

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
                        query_position * head_count_ +
                        head
                    ) * head_dimension();

                context_data[
                    context_offset + dimension
                ] = static_cast<float>(weighted_sum);
            }
        }
    }

    const Tensor merged_context =
        merge_heads(context);

    return output_projection_.forward(
        merged_context
    );
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