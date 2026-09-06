#pragma once

#include <cstddef>
#include <vector>

#include "edge_llm/tensor.hpp"

namespace edge_llm {

class Embedding {
public:
    explicit Embedding(Tensor weight);

    [[nodiscard]] Tensor forward(
        const std::vector<std::size_t>& token_ids
    ) const;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t embedding_dimension() const noexcept;
    [[nodiscard]] const Tensor& weight() const noexcept;

private:
    Tensor weight_;
};

class Linear {
public:
    explicit Linear(Tensor weight);

    [[nodiscard]] Tensor forward(
        const Tensor& input
    ) const;

    [[nodiscard]] std::size_t input_features() const noexcept;
    [[nodiscard]] std::size_t output_features() const noexcept;
    [[nodiscard]] const Tensor& weight() const noexcept;

private:
    Tensor weight_;
};

}  // namespace edge_llm