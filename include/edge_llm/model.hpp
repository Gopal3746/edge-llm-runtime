#pragma once

#include <cstddef>
#include <vector>

#include "edge_llm/kv_cache.hpp"
#include "edge_llm/layers.hpp"
#include "edge_llm/tensor.hpp"
#include "edge_llm/transformer.hpp"

namespace edge_llm {

class TransformerModel {
public:
    using LayerCaches = std::vector<KVCache>;

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

    [[nodiscard]] Tensor prefill(
        const std::vector<std::size_t>& token_ids,
        LayerCaches& caches
    ) const;

    [[nodiscard]] Tensor decode(
        std::size_t token_id,
        LayerCaches& caches
    ) const;

    [[nodiscard]] std::vector<std::size_t> generate(
        std::vector<std::size_t> prompt,
        std::size_t max_new_tokens
    ) const;

    [[nodiscard]] std::vector<std::size_t> generate_cached(
        std::vector<std::size_t> prompt,
        std::size_t max_new_tokens
    ) const;

    [[nodiscard]] LayerCaches create_caches(
        std::size_t capacity
    ) const;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t model_dimension() const noexcept;
    [[nodiscard]] std::size_t block_count() const noexcept;

private:
    void validate_cache_layout(
        const LayerCaches& caches
    ) const;

    [[nodiscard]] Tensor project_logits(
        const Tensor& hidden_states
    ) const;

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
