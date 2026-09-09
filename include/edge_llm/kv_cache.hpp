#pragma once

#include <cstddef>

#include "edge_llm/tensor.hpp"

namespace edge_llm {

class KVCache {
public:
    KVCache(
        std::size_t capacity,
        std::size_t head_count,
        std::size_t head_dimension
    );

    void append(
        const Tensor& keys,
        const Tensor& values
    );

    void clear() noexcept;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t remaining_capacity() const noexcept;

    [[nodiscard]] std::size_t head_count() const noexcept;
    [[nodiscard]] std::size_t head_dimension() const noexcept;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool full() const noexcept;

    [[nodiscard]] const float* key_data() const noexcept;
    [[nodiscard]] const float* value_data() const noexcept;

    [[nodiscard]] float key_at(
        std::size_t position,
        std::size_t head,
        std::size_t dimension
    ) const;

    [[nodiscard]] float value_at(
        std::size_t position,
        std::size_t head,
        std::size_t dimension
    ) const;

private:
    [[nodiscard]] std::size_t checked_offset(
        std::size_t position,
        std::size_t head,
        std::size_t dimension
    ) const;

    std::size_t capacity_;
    std::size_t head_count_;
    std::size_t head_dimension_;
    std::size_t size_;

    Tensor keys_;
    Tensor values_;
};

}  // namespace edge_llm
