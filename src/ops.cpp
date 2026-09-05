#include "edge_llm/ops.hpp"

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <string>

namespace edge_llm {
namespace {

void require_same_shape(
    const Tensor& left,
    const Tensor& right,
    std::string_view operation
) {
    if (left.shape() != right.shape()) {
        throw std::invalid_argument(
            "Tensor shapes must match for " +
            std::string(operation)
        );
    }
}

void require_matrix(
    const Tensor& tensor,
    std::string_view name
) {
    if (tensor.rank() != 2) {
        throw std::invalid_argument(
            std::string(name) +
            " must be a rank-2 tensor"
        );
    }
}

}  // namespace

Tensor add(
    const Tensor& left,
    const Tensor& right
) {
    require_same_shape(left, right, "addition");

    Tensor result(left.shape());

    const float* left_data = left.data();
    const float* right_data = right.data();
    float* result_data = result.data();

    for (std::size_t index = 0; index < left.size(); ++index) {
        result_data[index] =
            left_data[index] + right_data[index];
    }

    return result;
}

Tensor multiply(
    const Tensor& left,
    const Tensor& right
) {
    require_same_shape(
        left,
        right,
        "element-wise multiplication"
    );

    Tensor result(left.shape());

    const float* left_data = left.data();
    const float* right_data = right.data();
    float* result_data = result.data();

    for (std::size_t index = 0; index < left.size(); ++index) {
        result_data[index] =
            left_data[index] * right_data[index];
    }

    return result;
}

Tensor matmul(
    const Tensor& left,
    const Tensor& right
) {
    require_matrix(left, "Left operand");
    require_matrix(right, "Right operand");

    const std::size_t rows = left.shape()[0];
    const std::size_t shared_left = left.shape()[1];
    const std::size_t shared_right = right.shape()[0];
    const std::size_t columns = right.shape()[1];

    if (shared_left != shared_right) {
        throw std::invalid_argument(
            "Inner dimensions must match for matrix multiplication"
        );
    }

    Tensor result({rows, columns});

    const float* left_data = left.data();
    const float* right_data = right.data();
    float* result_data = result.data();

    for (std::size_t row = 0; row < rows; ++row) {
        for (
            std::size_t shared = 0;
            shared < shared_left;
            ++shared
        ) {
            const float left_value =
                left_data[row * shared_left + shared];

            for (
                std::size_t column = 0;
                column < columns;
                ++column
            ) {
                result_data[row * columns + column] +=
                    left_value *
                    right_data[shared * columns + column];
            }
        }
    }

    return result;
}

}  // namespace edge_llm