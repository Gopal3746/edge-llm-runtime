#include "edge_llm/layers.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace edge_llm {

Embedding::Embedding(Tensor weight)
    : weight_(std::move(weight)) {
    if (weight_.rank() != 2) {
        throw std::invalid_argument(
            "Embedding weight must be a rank-2 tensor"
        );
    }
}

Tensor Embedding::forward(
    const std::vector<std::size_t>& token_ids
) const {
    if (token_ids.empty()) {
        throw std::invalid_argument(
            "Embedding requires at least one token ID"
        );
    }

    const std::size_t dimension =
        embedding_dimension();

    Tensor output({
        token_ids.size(),
        dimension
    });

    const float* weight_data = weight_.data();
    float* output_data = output.data();

    for (
        std::size_t token_index = 0;
        token_index < token_ids.size();
        ++token_index
    ) {
        const std::size_t token_id =
            token_ids[token_index];

        if (token_id >= vocabulary_size()) {
            throw std::out_of_range(
                "Token ID is outside the embedding vocabulary"
            );
        }

        const float* source =
            weight_data + token_id * dimension;

        float* destination =
            output_data + token_index * dimension;

        std::copy_n(
            source,
            dimension,
            destination
        );
    }

    return output;
}

std::size_t Embedding::vocabulary_size() const noexcept {
    return weight_.shape()[0];
}

std::size_t Embedding::embedding_dimension() const noexcept {
    return weight_.shape()[1];
}

const Tensor& Embedding::weight() const noexcept {
    return weight_;
}

Linear::Linear(Tensor weight)
    : weight_(std::move(weight)) {
    if (weight_.rank() != 2) {
        throw std::invalid_argument(
            "Linear weight must be a rank-2 tensor"
        );
    }
}

Tensor Linear::forward(
    const Tensor& input
) const {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "Linear input must be a rank-2 tensor"
        );
    }

    const std::size_t row_count =
        input.shape()[0];

    const std::size_t provided_features =
        input.shape()[1];

    if (provided_features != input_features()) {
        throw std::invalid_argument(
            "Linear input size does not match its weight"
        );
    }

    Tensor output({
        row_count,
        output_features()
    });

    const float* input_data = input.data();
    const float* weight_data = weight_.data();
    float* output_data = output.data();

    for (
        std::size_t row = 0;
        row < row_count;
        ++row
    ) {
        const float* input_row =
            input_data + row * input_features();

        for (
            std::size_t output_index = 0;
            output_index < output_features();
            ++output_index
        ) {
            const float* weight_row =
                weight_data +
                output_index * input_features();

            double sum = 0.0;

            for (
                std::size_t input_index = 0;
                input_index < input_features();
                ++input_index
            ) {
                sum +=
                    static_cast<double>(
                        input_row[input_index]
                    ) *
                    static_cast<double>(
                        weight_row[input_index]
                    );
            }

            output_data[
                row * output_features() + output_index
            ] = static_cast<float>(sum);
        }
    }

    return output;
}

std::size_t Linear::input_features() const noexcept {
    return weight_.shape()[1];
}

std::size_t Linear::output_features() const noexcept {
    return weight_.shape()[0];
}

const Tensor& Linear::weight() const noexcept {
    return weight_;
}

}  // namespace edge_llm