#include "edge_llm/kv_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace edge_llm {
namespace {

Tensor make_cache_storage(
    std::size_t capacity,
    std::size_t head_count,
    std::size_t head_dimension
) {
    if (
        capacity == 0 ||
        head_count == 0 ||
        head_dimension == 0
    ) {
        throw std::invalid_argument(
            "KV cache dimensions must be greater than zero"
        );
    }

    return Tensor({
        capacity,
        head_count,
        head_dimension
    });
}

}  // namespace

KVCache::KVCache(
    std::size_t capacity,
    std::size_t head_count,
    std::size_t head_dimension
)
    : capacity_(capacity),
      head_count_(head_count),
      head_dimension_(head_dimension),
      size_(0),
      keys_(
          make_cache_storage(
              capacity,
              head_count,
              head_dimension
          )
      ),
      values_(
          make_cache_storage(
              capacity,
              head_count,
              head_dimension
          )
      ) {}

void KVCache::append(
    const Tensor& keys,
    const Tensor& values
) {
    if (keys.rank() != 3 || values.rank() != 3) {
        throw std::invalid_argument(
            "Cached keys and values must be rank-3 tensors"
        );
    }

    if (keys.shape() != values.shape()) {
        throw std::invalid_argument(
            "Cached key and value shapes must match"
        );
    }

    if (
        keys.shape()[1] != head_count_ ||
        keys.shape()[2] != head_dimension_
    ) {
        throw std::invalid_argument(
            "Cached tensors do not match cache dimensions"
        );
    }

    const std::size_t token_count =
        keys.shape()[0];

    if (token_count > remaining_capacity()) {
        throw std::length_error(
            "KV cache capacity exceeded"
        );
    }

    const std::size_t elements_per_token =
        head_count_ * head_dimension_;

    const std::size_t destination_offset =
        size_ * elements_per_token;

    std::copy_n(
        keys.data(),
        keys.size(),
        keys_.data() + destination_offset
    );

    std::copy_n(
        values.data(),
        values.size(),
        values_.data() + destination_offset
    );

    size_ += token_count;
}

void KVCache::clear() noexcept {
    size_ = 0;
}

std::size_t KVCache::size() const noexcept {
    return size_;
}

std::size_t KVCache::capacity() const noexcept {
    return capacity_;
}

std::size_t
KVCache::remaining_capacity() const noexcept {
    return capacity_ - size_;
}

std::size_t KVCache::head_count() const noexcept {
    return head_count_;
}

std::size_t KVCache::head_dimension() const noexcept {
    return head_dimension_;
}

bool KVCache::empty() const noexcept {
    return size_ == 0;
}

bool KVCache::full() const noexcept {
    return size_ == capacity_;
}

const float* KVCache::key_data() const noexcept {
    return keys_.data();
}

const float* KVCache::value_data() const noexcept {
    return values_.data();
}

float KVCache::key_at(
    std::size_t position,
    std::size_t head,
    std::size_t dimension
) const {
    return keys_.at(
        checked_offset(position, head, dimension)
    );
}

float KVCache::value_at(
    std::size_t position,
    std::size_t head,
    std::size_t dimension
) const {
    return values_.at(
        checked_offset(position, head, dimension)
    );
}

std::size_t KVCache::checked_offset(
    std::size_t position,
    std::size_t head,
    std::size_t dimension
) const {
    if (position >= size_) {
        throw std::out_of_range(
            "KV cache position is outside the populated range"
        );
    }

    if (head >= head_count_) {
        throw std::out_of_range(
            "KV cache head is outside the configured range"
        );
    }

    if (dimension >= head_dimension_) {
        throw std::out_of_range(
            "KV cache dimension is outside the configured range"
        );
    }

    return
        (
            position * head_count_ +
            head
        ) * head_dimension_ +
        dimension;
}

}  // namespace edge_llm
