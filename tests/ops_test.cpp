#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "edge_llm/ops.hpp"
#include "edge_llm/tensor.hpp"

namespace {

int failure_count = 0;

void expect(
    bool condition,
    std::string_view message
) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failure_count;
    }
}

bool nearly_equal(
    float left,
    float right,
    float tolerance = 1.0e-5F
) {
    return std::fabs(left - right) <= tolerance;
}

void expect_values(
    const edge_llm::Tensor& tensor,
    const std::vector<float>& expected,
    std::string_view message
) {
    if (tensor.size() != expected.size()) {
        expect(false, message);
        return;
    }

    for (
        std::size_t index = 0;
        index < expected.size();
        ++index
    ) {
        if (!nearly_equal(tensor.at(index), expected[index])) {
            expect(false, message);
            return;
        }
    }

    expect(true, message);
}

template <typename Function>
void expect_invalid_argument(
    Function function,
    std::string_view message
) {
    bool threw_expected_exception = false;

    try {
        function();
    } catch (const std::invalid_argument&) {
        threw_expected_exception = true;
    } catch (...) {
        // A different exception does not satisfy the test.
    }

    expect(threw_expected_exception, message);
}

}  // namespace

int main() {
    using edge_llm::Tensor;
    using edge_llm::add;
    using edge_llm::matmul;
    using edge_llm::multiply;

    const Tensor first(
        {2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F}
    );

    const Tensor second(
        {2, 2},
        {5.0F, 6.0F, 7.0F, 8.0F}
    );

    const Tensor sum = add(first, second);

    expect(
        sum.shape() == Tensor::Shape({2, 2}),
        "Addition should preserve the input shape"
    );

    expect_values(
        sum,
        {6.0F, 8.0F, 10.0F, 12.0F},
        "Addition should add corresponding elements"
    );

    const Tensor product = multiply(first, second);

    expect_values(
        product,
        {5.0F, 12.0F, 21.0F, 32.0F},
        "Multiplication should multiply corresponding elements"
    );

    const Tensor matrix_left(
        {2, 3},
        {
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F
        }
    );

    const Tensor matrix_right(
        {3, 2},
        {
            7.0F, 8.0F,
            9.0F, 10.0F,
            11.0F, 12.0F
        }
    );

    const Tensor matrix_result =
        matmul(matrix_left, matrix_right);

    expect(
        matrix_result.shape() == Tensor::Shape({2, 2}),
        "Matrix multiplication should produce an M by N matrix"
    );

    expect_values(
        matrix_result,
        {58.0F, 64.0F, 139.0F, 154.0F},
        "Matrix multiplication should calculate correct values"
    );

    expect_invalid_argument(
        [&first] {
            const Tensor wrong_shape({4}, {1.0F, 2.0F, 3.0F, 4.0F});
            static_cast<void>(add(first, wrong_shape));
        },
        "Addition should reject mismatched shapes"
    );

    expect_invalid_argument(
        [&first] {
            const Tensor wrong_shape({1, 4}, {1.0F, 2.0F, 3.0F, 4.0F});
            static_cast<void>(multiply(first, wrong_shape));
        },
        "Element-wise multiplication should reject mismatched shapes"
    );

    expect_invalid_argument(
        [&matrix_right] {
            const Tensor vector(
                {3},
                {1.0F, 2.0F, 3.0F}
            );

            static_cast<void>(matmul(vector, matrix_right));
        },
        "Matrix multiplication should require rank-2 tensors"
    );

    expect_invalid_argument(
        [] {
            const Tensor left({2, 3});
            const Tensor right({2, 2});

            static_cast<void>(matmul(left, right));
        },
        "Matrix multiplication should reject incompatible dimensions"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " tensor operation test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All tensor operation tests passed\n";
    return EXIT_SUCCESS;
}