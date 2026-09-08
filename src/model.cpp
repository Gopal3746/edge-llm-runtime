#include "edge_llm/model.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "edge_llm/nn.hpp"

namespace edge_llm {

TransformerModel::TransformerModel(
    Embedding token_embedding,
    std::vector<TransformerBlock> blocks,
    Tensor final_norm_weight,
    Linear output_projection,
    float norm_epsilon
)
    : token_embedding_(std::move(token_embedding)),
      blocks_(std::move(blocks)),
      final_norm_weight_(std::move(final_norm_weight)),
      output_projection_(std::move(output_projection)),
      norm_epsilon_(norm_epsilon) {
    if (blocks_.empty()) {
        throw std::invalid_argument(
            "Transformer model requires at least one block"
        );
    }

    if (
        !std::isfinite(norm_epsilon_) ||
        norm_epsilon_ <= 0.0F
    ) {
        throw std::invalid_argument(
            "Model norm epsilon must be finite and positive"
        );
    }

    const std::size_t dimension =
        token_embedding_.embedding_dimension();

    for (const TransformerBlock& block : blocks_) {
        if (block.model_dimension() != dimension) {
            throw std::invalid_argument(
                "Every transformer block must match "
                "the embedding dimension"
            );
        }
    }

    if (
        final_norm_weight_.rank() != 1 ||
        final_norm_weight_.size() != dimension
    ) {
        throw std::invalid_argument(
            "Final norm weight must match model dimension"
        );
    }

    if (
        output_projection_.input_features() != dimension ||
        output_projection_.output_features() !=
            token_embedding_.vocabulary_size()
    ) {
        throw std::invalid_argument(
            "Output projection must map model dimension "
            "to vocabulary size"
        );
    }
}

Tensor TransformerModel::forward(
    const std::vector<std::size_t>& token_ids
) const {
    if (token_ids.empty()) {
        throw std::invalid_argument(
            "Model input must contain at least one token"
        );
    }

    Tensor hidden_states =
        token_embedding_.forward(token_ids);

    for (const TransformerBlock& block : blocks_) {
        hidden_states =
            block.forward(hidden_states);
    }

    const Tensor normalized =
        rms_norm(
            hidden_states,
            final_norm_weight_,
            norm_epsilon_
        );

    return output_projection_.forward(normalized);
}

std::vector<std::size_t> TransformerModel::generate(
    std::vector<std::size_t> prompt,
    std::size_t max_new_tokens
) const {
    if (prompt.empty()) {
        throw std::invalid_argument(
            "Generation prompt must contain at least one token"
        );
    }

    if (
        max_new_tokens >
        prompt.max_size() - prompt.size()
    ) {
        throw std::length_error(
            "Requested token count exceeds vector capacity"
        );
    }

    prompt.reserve(
        prompt.size() + max_new_tokens
    );

    for (
        std::size_t step = 0;
        step < max_new_tokens;
        ++step
    ) {
        const Tensor logits = forward(prompt);

        const std::size_t next_token =
            select_greedy_token(logits);

        prompt.push_back(next_token);
    }

    return prompt;
}

std::size_t
TransformerModel::select_greedy_token(
    const Tensor& logits
) const {
    if (
        logits.rank() != 2 ||
        logits.shape()[1] != vocabulary_size()
    ) {
        throw std::invalid_argument(
            "Logits must have shape [sequence, vocabulary]"
        );
    }

    const std::size_t sequence_length =
        logits.shape()[0];

    const std::size_t last_row_offset =
        (sequence_length - 1) * vocabulary_size();

    std::size_t best_token = 0;
    float best_value = logits.at(last_row_offset);

    for (
        std::size_t token = 1;
        token < vocabulary_size();
        ++token
    ) {
        const float value =
            logits.at(last_row_offset + token);

        if (value > best_value) {
            best_value = value;
            best_token = token;
        }
    }

    return best_token;
}

std::size_t
TransformerModel::vocabulary_size() const noexcept {
    return token_embedding_.vocabulary_size();
}

std::size_t
TransformerModel::model_dimension() const noexcept {
    return token_embedding_.embedding_dimension();
}

std::size_t
TransformerModel::block_count() const noexcept {
    return blocks_.size();
}

}  // namespace edge_llm