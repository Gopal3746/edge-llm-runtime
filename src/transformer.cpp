#include "edge_llm/transformer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "edge_llm/nn.hpp"
#include "edge_llm/ops.hpp"

namespace edge_llm {
namespace {

void validate_norm_weight(
    const Tensor& weight,
    std::size_t model_dimension,
    std::string_view name
) {
    if (
        weight.rank() != 1 ||
        weight.size() != model_dimension
    ) {
        throw std::invalid_argument(
            std::string(name) +
            " must be rank 1 and match model dimension"
        );
    }
}

}  // namespace

TransformerBlock::TransformerBlock(
    CausalSelfAttention attention,
    SwiGLUFeedForward feed_forward,
    Tensor attention_norm_weight,
    Tensor feed_forward_norm_weight,
    float norm_epsilon
)
    : attention_(std::move(attention)),
      feed_forward_(std::move(feed_forward)),
      attention_norm_weight_(
          std::move(attention_norm_weight)
      ),
      feed_forward_norm_weight_(
          std::move(feed_forward_norm_weight)
      ),
      norm_epsilon_(norm_epsilon) {
    if (
        !std::isfinite(norm_epsilon_) ||
        norm_epsilon_ <= 0.0F
    ) {
        throw std::invalid_argument(
            "Transformer norm epsilon must be finite and positive"
        );
    }

    if (
        attention_.model_dimension() !=
        feed_forward_.model_dimension()
    ) {
        throw std::invalid_argument(
            "Attention and feed-forward model dimensions must match"
        );
    }

    validate_norm_weight(
        attention_norm_weight_,
        model_dimension(),
        "Attention norm weight"
    );

    validate_norm_weight(
        feed_forward_norm_weight_,
        model_dimension(),
        "Feed-forward norm weight"
    );
}

void TransformerBlock::validate_input(
    const Tensor& input
) const {
    if (
        input.rank() != 2 ||
        input.shape()[1] != model_dimension()
    ) {
        throw std::invalid_argument(
            "Transformer input must have shape "
            "[sequence, model dimension]"
        );
    }
}

Tensor TransformerBlock::finish_block(
    const Tensor& input,
    const Tensor& attention_output
) const {
    const Tensor attention_residual =
        add(input, attention_output);

    const Tensor normalized_feed_forward_input =
        rms_norm(
            attention_residual,
            feed_forward_norm_weight_,
            norm_epsilon_
        );

    const Tensor feed_forward_output =
        feed_forward_.forward(
            normalized_feed_forward_input
        );

    return add(
        attention_residual,
        feed_forward_output
    );
}

Tensor TransformerBlock::forward(
    const Tensor& input,
    std::size_t position_offset
) const {
    validate_input(input);

    const Tensor normalized_attention_input =
        rms_norm(
            input,
            attention_norm_weight_,
            norm_epsilon_
        );

    const Tensor attention_output =
        attention_.forward(
            normalized_attention_input,
            position_offset
        );

    return finish_block(
        input,
        attention_output
    );
}

Tensor TransformerBlock::prefill(
    const Tensor& input,
    KVCache& cache
) const {
    validate_input(input);

    const Tensor normalized_attention_input =
        rms_norm(
            input,
            attention_norm_weight_,
            norm_epsilon_
        );

    const Tensor attention_output =
        attention_.prefill(
            normalized_attention_input,
            cache
        );

    return finish_block(
        input,
        attention_output
    );
}

Tensor TransformerBlock::decode(
    const Tensor& input,
    KVCache& cache
) const {
    validate_input(input);

    if (input.shape()[0] != 1) {
        throw std::invalid_argument(
            "Transformer decode input must contain exactly one token"
        );
    }

    const Tensor normalized_attention_input =
        rms_norm(
            input,
            attention_norm_weight_,
            norm_epsilon_
        );

    const Tensor attention_output =
        attention_.decode(
            normalized_attention_input,
            cache
        );

    return finish_block(
        input,
        attention_output
    );
}

std::size_t
TransformerBlock::model_dimension() const noexcept {
    return attention_.model_dimension();
}

std::size_t
TransformerBlock::hidden_dimension() const noexcept {
    return feed_forward_.hidden_dimension();
}

std::size_t
TransformerBlock::head_count() const noexcept {
    return attention_.head_count();
}

std::size_t
TransformerBlock::head_dimension() const noexcept {
    return attention_.head_dimension();
}

}  // namespace edge_llm
