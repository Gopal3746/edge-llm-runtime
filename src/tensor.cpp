#include "edge_llm/tensor.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace edge_llm {

Tensor::Tensor(Shape shape)
    : shape_(std::move(shape)),
      data_(checked_element_count(shape_), 0.0F) {}

Tensor::Tensor(
    Shape shape,
    std::vector<float> values
)
    : shape_(std::move(shape)),
      data_(std::move(values)) {
    const std::size_t expected_size =
        checked_element_count(shape_);

    if (data_.size() != expected_size) {
        throw std::invalid_argument(
            "Tensor data size does not match its shape"
        );
    }
}

const Tensor::Shape& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

std::size_t Tensor::size() const noexcept {
    return data_.size();
}

bool Tensor::empty() const noexcept {
    return data_.empty();
}

float* Tensor::data() noexcept {
    return data_.data();
}

const float* Tensor::data() const noexcept {
    return data_.data();
}

float& Tensor::at(std::size_t flat_index) {
    return data_.at(flat_index);
}

const float& Tensor::at(std::size_t flat_index) const {
    return data_.at(flat_index);
}

float& Tensor::at(const Shape& indices) {
    return data_.at(flatten_index(indices));
}

const float& Tensor::at(const Shape& indices) const {
    return data_.at(flatten_index(indices));
}

std::size_t Tensor::checked_element_count(
    const Shape& shape
) {
    if (shape.empty()) {
        throw std::invalid_argument(
            "Tensor shape must contain at least one dimension"
        );
    }

    std::size_t element_count = 1;

    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be greater than zero"
            );
        }

        if (
            element_count >
            std::numeric_limits<std::size_t>::max() / dimension
        ) {
            throw std::overflow_error(
                "Tensor element count overflow"
            );
        }

        element_count *= dimension;
    }

    return element_count;
}

std::size_t Tensor::flatten_index(
    const Shape& indices
) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument(
            "Tensor index rank does not match tensor rank"
        );
    }

    std::size_t flat_index = 0;

    for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
        if (indices[axis] >= shape_[axis]) {
            throw std::out_of_range(
                "Tensor index is outside its shape"
            );
        }

        flat_index =
            flat_index * shape_[axis] + indices[axis];
    }

    return flat_index;
}

}  // namespace edge_llm