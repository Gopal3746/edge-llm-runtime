#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "edge_llm/rope.hpp"
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
        // A different exception does not satisfy the test.
    }

    expect(threw_expected_exception, message);
}

}  // namespace

int main() {
    using edge_llm::Tensor;
    using edge_llm::apply_rope;

    const Tensor input(
        {2, 1, 4},
        {
            1.0F, 2.0F, 3.0F, 4.0F,
            1.0F, 0.0F, 0.0F, 1.0F
        }
    );

    const Tensor rotated = apply_rope(input);

    expect(
        rotated.shape() == input.shape(),
        "RoPE should preserve the input shape"
    );

    expect(
        nearly_equal(rotated.at(0), 1.0F) &&
        nearly_equal(rotated.at(1), 2.0F) &&
        nearly_equal(rotated.at(2), 3.0F) &&
        nearly_equal(rotated.at(3), 4.0F),
        "RoPE should leave position zero unchanged"
    );

    expect(
        nearly_equal(
            rotated.at(4),
            static_cast<float>(std::cos(1.0))
        ),
        "RoPE should rotate the first pair using position one"
    );

    expect(
        nearly_equal(
            rotated.at(5),
            static_cast<float>(std::sin(1.0))
        ),
        "RoPE should calculate the rotated second coordinate"
    );

    constexpr double second_pair_angle = 0.01;

    expect(
        nearly_equal(
            rotated.at(6),
            static_cast<float>(
                -std::sin(second_pair_angle)
            )
        ),
        "RoPE should use a lower frequency for later pairs"
    );

    expect(
        nearly_equal(
            rotated.at(7),
            static_cast<float>(
                std::cos(second_pair_angle)
            )
        ),
        "RoPE should rotate every feature pair"
    );

    const float original_pair_norm =
        input.at(4) * input.at(4) +
        input.at(5) * input.at(5);

    const float rotated_pair_norm =
        rotated.at(4) * rotated.at(4) +
        rotated.at(5) * rotated.at(5);

    expect(
        nearly_equal(original_pair_norm, rotated_pair_norm),
        "Rotation should preserve each pair's squared magnitude"
    );

    const Tensor offset_input(
        {1, 1, 2},
        {1.0F, 0.0F}
    );

    const Tensor offset_rotated =
        apply_rope(offset_input, 2);

    expect(
        nearly_equal(
            offset_rotated.at(0),
            static_cast<float>(std::cos(2.0))
        ) &&
        nearly_equal(
            offset_rotated.at(1),
            static_cast<float>(std::sin(2.0))
        ),
        "RoPE should include the supplied position offset"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const Tensor rank_two(
                {2, 4},
                {
                    1.0F, 2.0F, 3.0F, 4.0F,
                    5.0F, 6.0F, 7.0F, 8.0F
                }
            );

            static_cast<void>(apply_rope(rank_two));
        },
        "RoPE should require a rank-3 input"
    );

    expect_throws<std::invalid_argument>(
        [] {
            const Tensor odd_dimension(
                {1, 1, 3},
                {1.0F, 2.0F, 3.0F}
            );

            static_cast<void>(
                apply_rope(odd_dimension)
            );
        },
        "RoPE should reject an odd head dimension"
    );

    expect_throws<std::invalid_argument>(
        [&input] {
            static_cast<void>(
                apply_rope(input, 0, 0.0)
            );
        },
        "RoPE should reject a non-positive base"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " RoPE test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout << "All RoPE tests passed\n";
    return EXIT_SUCCESS;
}
