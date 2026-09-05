#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

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

template <typename Exception, typename Function>
void expect_throws(
    Function function,
    std::string_view message
) {
    bool threw_expected_exception = false;

    try {
        function();
    } catch (const Exception&) {
        threw_expected_exception = true;
    } catch (...) {
        // A different exception type does not satisfy the test.
    }

    expect(threw_expected_exception, message);
}

}  // namespace

int main() {
    using edge_llm::Tensor;

    Tensor zeros({2, 3});

    expect(
        zeros.shape() == Tensor::Shape({2, 3}),
        "Tensor should preserve its shape"
    );

    expect(
        zeros.rank() == 2,
        "Tensor rank should equal the number of dimensions"
    );

    expect(
        zeros.size() == 6,
        "Tensor size should equal the product of its dimensions"
    );

    expect(
        !zeros.empty(),
        "A valid tensor should not be empty"
    );

    for (std::size_t index = 0; index < zeros.size(); ++index) {
        expect(
            zeros.at(index) == 0.0F,
            "Shape-only construction should initialize values to zero"
        );
    }

    Tensor values(
        {2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F}
    );

    expect(
        values.at(Tensor::Shape{0, 0}) == 1.0F,
        "Coordinate [0, 0] should map to the first value"
    );

    expect(
        values.at(Tensor::Shape{1, 0}) == 3.0F,
        "Coordinate [1, 0] should use row-major indexing"
    );

    expect(
        values.at(Tensor::Shape{1, 1}) == 4.0F,
        "Coordinate [1, 1] should map to the final value"
    );

    values.at(Tensor::Shape{0, 1}) = 9.0F;

    expect(
        values.at(1) == 9.0F,
        "Multidimensional writes should update flat storage"
    );

    expect(
        values.data()[2] == 3.0F,
        "Tensor data should be stored contiguously"
    );

    expect_throws<std::invalid_argument>(
        [] {
            Tensor invalid(Tensor::Shape{});
        },
        "An empty shape should be rejected"
    );

    expect_throws<std::invalid_argument>(
        [] {
            Tensor invalid({2, 0});
        },
        "A zero-sized dimension should be rejected"
    );

    expect_throws<std::invalid_argument>(
        [] {
            Tensor invalid(
                {2, 3},
                {1.0F, 2.0F}
            );
        },
        "A mismatched value count should be rejected"
    );

    expect_throws<std::invalid_argument>(
        [&values] {
            values.at(Tensor::Shape{0});
        },
        "An index with the wrong rank should be rejected"
    );

    expect_throws<std::out_of_range>(
        [&values] {
            values.at(Tensor::Shape{2, 0});
        },
        "An out-of-range coordinate should be rejected"
    );

    expect_throws<std::out_of_range>(
        [&values] {
            values.at(4);
        },
        "An out-of-range flat index should be rejected"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " tensor test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All tensor tests passed\n";
    return EXIT_SUCCESS;
}