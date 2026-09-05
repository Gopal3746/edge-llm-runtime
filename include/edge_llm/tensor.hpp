#pragma once

#include <cstddef>
#include <vector>

namespace edge_llm {

class Tensor {
public:
    using Shape = std::vector<std::size_t>;

    explicit Tensor(Shape shape);

    Tensor(
        Shape shape,
        std::vector<float> values
    );

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] float* data() noexcept;
    [[nodiscard]] const float* data() const noexcept;

    float& at(std::size_t flat_index);
    const float& at(std::size_t flat_index) const;

    float& at(const Shape& indices);
    const float& at(const Shape& indices) const;

private:
    [[nodiscard]] static std::size_t checked_element_count(
        const Shape& shape
    );

    [[nodiscard]] std::size_t flatten_index(
        const Shape& indices
    ) const;

    Shape shape_;
    std::vector<float> data_;
};

}  // namespace edge_llm