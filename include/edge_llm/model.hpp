#pragma once

#include <cstddef>
#include <vector>

#include "edge_llm/layers.hpp"
#include "edge_llm/tensor.hpp"
#include "edge_llm/transformer.hpp"

namespace edge_llm {

class TransformerModel {
public:
    TransformerModel(
        Embedding token_embedding,
        std::vector<TransformerBlock> blocks,
        Tensor final_norm_weight,
        Linear output_projection,
        float norm_epsilon = 1.0e-5F
    );

    [[nodiscard]] Tensor forward(
        const std::vector<std::size_t>& token_ids
    ) const;

    [[nodiscard]] std::vector<std::size_t> generate(
        std::vector<std::size_t> prompt,
        std::size_t max_new_tokens
    ) const;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t model_dimension() const noexcept;
    [[nodiscard]] std::size_t block_count() const noexcept;

private:
    [[nodiscard]] std::size_t select_greedy_token(
        const Tensor& logits
    ) const;

    Embedding token_embedding_;
    std::vector<TransformerBlock> blocks_;
    Tensor final_norm_weight_;
    Linear output_projection_;
    float norm_epsilon_;
};

}  // namespace edge_llm