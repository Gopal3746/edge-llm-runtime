#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "edge_llm/quantization.hpp"
#include "edge_llm/tensor.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct TimingStats {
    double mean_ms;
    double median_ms;
    double p95_ms;
    double minimum_ms;
    double maximum_ms;
};

edge_llm::Tensor random_tensor(
    edge_llm::Tensor::Shape shape,
    std::mt19937& generator,
    float magnitude
) {
    std::uniform_real_distribution<float> distribution(
        -magnitude,
        magnitude
    );

    edge_llm::Tensor tensor(
        std::move(shape)
    );

    for (
        std::size_t index = 0;
        index < tensor.size();
        ++index
    ) {
        tensor.at(index) =
            distribution(generator);
    }

    return tensor;
}

TimingStats summarize(
    std::vector<double> durations
) {
    if (durations.empty()) {
        throw std::invalid_argument(
            "Timing results must not be empty"
        );
    }

    std::sort(
        durations.begin(),
        durations.end()
    );

    double total = 0.0;

    for (const double duration : durations) {
        total += duration;
    }

    const std::size_t count =
        durations.size();

    const double median =
        count % 2 == 0
            ? (
                durations[count / 2 - 1] +
                durations[count / 2]
            ) / 2.0
            : durations[count / 2];

    const std::size_t p95_index =
        std::min(
            count - 1,
            static_cast<std::size_t>(
                std::ceil(
                    0.95 *
                    static_cast<double>(count)
                )
            ) - 1
        );

    return TimingStats{
        total / static_cast<double>(count),
        median,
        durations[p95_index],
        durations.front(),
        durations.back()
    };
}

template <typename Operation>
double measure_once(
    Operation operation,
    double& checksum
) {
    const auto start = Clock::now();

    checksum += operation();

    const auto end = Clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

void print_stats(
    const char* prefix,
    const TimingStats& stats
) {
    std::cout
        << prefix << "_mean_ms="
        << stats.mean_ms << '\n'
        << prefix << "_median_ms="
        << stats.median_ms << '\n'
        << prefix << "_p95_ms="
        << stats.p95_ms << '\n'
        << prefix << "_min_ms="
        << stats.minimum_ms << '\n'
        << prefix << "_max_ms="
        << stats.maximum_ms << '\n';
}

}  // namespace

int main() {
    try {
        constexpr std::size_t random_seed = 42;
        constexpr std::size_t rows = 1;
        constexpr std::size_t input_features = 256;
        constexpr std::size_t output_features = 256;
        constexpr std::size_t warmup_iterations = 100;
        constexpr std::size_t measured_iterations = 2000;

        std::mt19937 generator(random_seed);

        const edge_llm::Tensor weight =
            random_tensor(
                {
                    output_features,
                    input_features
                },
                generator,
                0.25F
            );

        const edge_llm::Tensor input =
            random_tensor(
                {
                    rows,
                    input_features
                },
                generator,
                1.0F
            );

        const edge_llm::QuantizedLinear linear(
            weight
        );

        edge_llm::Tensor reusable_output({
            rows,
            output_features
        });

        const edge_llm::Tensor reference_output =
            linear.forward(input);

        linear.forward_into(
            input,
            reusable_output
        );

        for (
            std::size_t index = 0;
            index < reference_output.size();
            ++index
        ) {
            if (
                reference_output.at(index) !=
                reusable_output.at(index)
            ) {
                throw std::runtime_error(
                    "Reusable and allocating outputs differ"
                );
            }
        }

        double checksum = 0.0;

        for (
            std::size_t iteration = 0;
            iteration < warmup_iterations;
            ++iteration
        ) {
            const edge_llm::Tensor output =
                linear.forward(input);

            checksum += output.at(
                output.size() - 1
            );

            linear.forward_into(
                input,
                reusable_output
            );

            checksum += reusable_output.at(
                reusable_output.size() - 1
            );
        }

        std::vector<double> allocating_durations;
        std::vector<double> reused_durations;

        allocating_durations.reserve(
            measured_iterations
        );

        reused_durations.reserve(
            measured_iterations
        );

        const auto allocating_operation =
            [&linear, &input] {
                const edge_llm::Tensor output =
                    linear.forward(input);

                return static_cast<double>(
                    output.at(output.size() - 1)
                );
            };

        const auto reused_operation =
            [&linear, &input, &reusable_output] {
                linear.forward_into(
                    input,
                    reusable_output
                );

                return static_cast<double>(
                    reusable_output.at(
                        reusable_output.size() - 1
                    )
                );
            };

        for (
            std::size_t iteration = 0;
            iteration < measured_iterations;
            ++iteration
        ) {
            if (iteration % 2 == 0) {
                allocating_durations.push_back(
                    measure_once(
                        allocating_operation,
                        checksum
                    )
                );

                reused_durations.push_back(
                    measure_once(
                        reused_operation,
                        checksum
                    )
                );
            } else {
                reused_durations.push_back(
                    measure_once(
                        reused_operation,
                        checksum
                    )
                );

                allocating_durations.push_back(
                    measure_once(
                        allocating_operation,
                        checksum
                    )
                );
            }
        }

        const TimingStats allocating_stats =
            summarize(
                std::move(allocating_durations)
            );

        const TimingStats reused_stats =
            summarize(
                std::move(reused_durations)
            );

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::cout
            << "edge_llm_buffer_benchmark_version=1\n"
            << "mode=single_token_decode_like_linear\n"
            << "random_seed=" << random_seed << '\n'
            << "rows=" << rows << '\n'
            << "input_features="
            << input_features << '\n'
            << "output_features="
            << output_features << '\n'
            << "warmup_iterations="
            << warmup_iterations << '\n'
            << "measured_iterations="
            << measured_iterations << '\n'
            << "reusable_buffer_bytes="
            << reusable_output.size() *
                sizeof(float) << '\n'
            << "output_allocations_avoided_per_call=1\n";

        print_stats(
            "allocating",
            allocating_stats
        );

        print_stats(
            "reused",
            reused_stats
        );

        std::cout
            << "reused_over_allocating_speedup="
            << allocating_stats.mean_ms /
                reused_stats.mean_ms << '\n'
            << "mean_latency_reduction_percent="
            << (
                1.0 -
                reused_stats.mean_ms /
                    allocating_stats.mean_ms
            ) * 100.0 << '\n'
            << "benchmark_checksum="
            << checksum << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr
            << "Buffer benchmark failed: "
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}
