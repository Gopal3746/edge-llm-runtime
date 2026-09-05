#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "edge_llm/nn.hpp"
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
        // Another exception type does not satisfy the test.
    }

    expect(threw_expected_exception, message);
}

}  // namespace

int main() {
    using edge_llm::Tensor;
    using edge_llm::rms_norm;
    using edge_llm::softmax_last_dim;

    constexpr float epsilon = 1.0e-5F;

    const Tensor rms_input(
        {2, 2},
        {
            3.0F, 4.0F,
            0.0F, 5.0F
        }
    );

    const Tensor rms_weight(
        {2},
        {1.0F, 2.0F}
    );

    const Tensor normalized =
        rms_norm(rms_input, rms_weight, epsilon);

    expect(
        normalized.shape() == rms_input.shape(),
        "RMSNorm should preserve the input shape"
    );

    const float first_inverse_rms =
        1.0F / std::sqrt(12.5F + epsilon);

    expect(
        nearly_equal(
            normalized.at(0),
            3.0F * first_inverse_rms
        ),
        "RMSNorm should normalize the first value"
    );

    expect(
        nearly_equal(
            normalized.at(1),
            4.0F * first_inverse_rms * 2.0F
        ),
        "RMSNorm should apply its learned weight"
    );

    const float second_inverse_rms =
        1.0F / std::sqrt(12.5F + epsilon);

    expect(
        nearly_equal(normalized.at(2), 0.0F),
        "RMSNorm should preserve a zero activation"
    );

    expect(
        nearly_equal(
            normalized.at(3),
            5.0F * second_inverse_rms * 2.0F
        ),
        "RMSNorm should normalize rows independently"
    );

    const Tensor logits(
        {2, 3},
        {
            1.0F, 2.0F, 3.0F,
            1000.0F, 1000.0F, 1000.0F
        }
    );

    const Tensor probabilities =
        softmax_last_dim(logits);

    expect(
        probabilities.shape() == logits.shape(),
        "Softmax should preserve the input shape"
    );

    expect(
        nearly_equal(probabilities.at(0), 0.0900306F),
        "Softmax should calculate the first probability"
    );

    expect(
        nearly_equal(probabilities.at(1), 0.244728F),
        "Softmax should calculate the second probability"
    );

    expect(
        nearly_equal(probabilities.at(2), 0.665241F),
        "Softmax should calculate the third probability"
    );

    const float first_row_sum =
        probabilities.at(0) +
        probabilities.at(1) +
        probabilities.at(2);

    expect(
        nearly_equal(first_row_sum, 1.0F),
        "Each softmax row should sum to one"
    );

    expect(
        nearly_equal(probabilities.at(3), 1.0F / 3.0F) &&
        nearly_equal(probabilities.at(4), 1.0F / 3.0F) &&
        nearly_equal(probabilities.at(5), 1.0F / 3.0F),
        "Softmax should remain stable for large logits"
    );

    expect_invalid_argument(
        [&rms_input, &rms_weight] {
            static_cast<void>(
                rms_norm(rms_input, rms_weight, 0.0F)
            );
        },
        "RMSNorm should reject a non-positive epsilon"
    );

    expect_invalid_argument(
        [&rms_input] {
            const Tensor wrong_rank_weight(
                {1, 2},
                {1.0F, 1.0F}
            );

            static_cast<void>(
                rms_norm(rms_input, wrong_rank_weight)
            );
        },
        "RMSNorm should require a rank-1 weight"
    );

    expect_invalid_argument(
        [&rms_input] {
            const Tensor wrong_size_weight(
                {3},
                {1.0F, 1.0F, 1.0F}
            );

            static_cast<void>(
                rms_norm(rms_input, wrong_size_weight)
            );
        },
        "RMSNorm should reject an incompatible weight size"
    );

    if (failure_count != 0) {
        std::cerr
            << failure_count
            << " neural-network operation test(s) failed\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "All neural-network operation tests passed\n";

    return EXIT_SUCCESS;
}