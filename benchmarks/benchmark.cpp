#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "edge_llm/attention.hpp"
#include "edge_llm/feed_forward.hpp"
#include "edge_llm/layers.hpp"
#include "edge_llm/model.hpp"
#include "edge_llm/quantization.hpp"
#include "edge_llm/tensor.hpp"
#include "edge_llm/transformer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct TimingStats {
    double mean_ms;
    double median_ms;
    double p95_ms;
    double minimum_ms;
    double maximum_ms;
    double checksum;
};

struct ErrorMetrics {
    double maximum_absolute_error;
    double mean_absolute_error;
};

TimingStats summarize(
    std::vector<double> durations,
    double checksum
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

    const std::size_t count = durations.size();

    double median = 0.0;

    if (count % 2 == 0) {
        median =
            (
                durations[count / 2 - 1] +
                durations[count / 2]
            ) / 2.0;
    } else {
        median = durations[count / 2];
    }

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
        durations.back(),
        checksum
    };
}

template <
    typename Operation,
    typename Consumer
>
TimingStats measure(
    Operation operation,
    Consumer consume,
    std::size_t warmup_iterations,
    std::size_t measured_iterations
) {
    double checksum = 0.0;

    for (
        std::size_t iteration = 0;
        iteration < warmup_iterations;
        ++iteration
    ) {
        const auto output = operation();
        checksum += consume(output);
    }

    std::vector<double> durations;
    durations.reserve(measured_iterations);

    for (
        std::size_t iteration = 0;
        iteration < measured_iterations;
        ++iteration
    ) {
        const auto start = Clock::now();

        const auto output = operation();

        const auto end = Clock::now();

        checksum += consume(output);

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        durations.push_back(elapsed_ms);
    }

    return summarize(
        std::move(durations),
        checksum
    );
}

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

edge_llm::Tensor ones(
    std::size_t size
) {
    return edge_llm::Tensor(
        {size},
        std::vector<float>(size, 1.0F)
    );
}

edge_llm::TransformerBlock make_block(
    std::mt19937& generator,
    std::size_t model_dimension,
    std::size_t hidden_dimension,
    std::size_t head_count
) {
    edge_llm::Tensor query_weight =
        random_tensor(
            {model_dimension, model_dimension},
            generator,
            0.05F
        );

    edge_llm::Tensor key_weight =
        random_tensor(
            {model_dimension, model_dimension},
            generator,
            0.05F
        );

    edge_llm::Tensor value_weight =
        random_tensor(
            {model_dimension, model_dimension},
            generator,
            0.05F
        );

    edge_llm::Tensor attention_output_weight =
        random_tensor(
            {model_dimension, model_dimension},
            generator,
            0.05F
        );

    edge_llm::CausalSelfAttention attention(
        std::move(query_weight),
        std::move(key_weight),
        std::move(value_weight),
        std::move(attention_output_weight),
        head_count
    );

    edge_llm::Tensor gate_weight =
        random_tensor(
            {hidden_dimension, model_dimension},
            generator,
            0.05F
        );

    edge_llm::Tensor up_weight =
        random_tensor(
            {hidden_dimension, model_dimension},
            generator,
            0.05F
        );

    edge_llm::Tensor down_weight =
        random_tensor(
            {model_dimension, hidden_dimension},
            generator,
            0.05F
        );

    edge_llm::SwiGLUFeedForward feed_forward(
        std::move(gate_weight),
        std::move(up_weight),
        std::move(down_weight)
    );

    return edge_llm::TransformerBlock(
        std::move(attention),
        std::move(feed_forward),
        ones(model_dimension),
        ones(model_dimension)
    );
}

edge_llm::TransformerModel make_model(
    std::mt19937& generator,
    std::size_t vocabulary_size,
    std::size_t model_dimension,
    std::size_t hidden_dimension,
    std::size_t head_count,
    std::size_t block_count
) {
    edge_llm::Embedding embedding(
        random_tensor(
            {vocabulary_size, model_dimension},
            generator,
            0.1F
        )
    );

    std::vector<edge_llm::TransformerBlock> blocks;
    blocks.reserve(block_count);

    for (
        std::size_t block = 0;
        block < block_count;
        ++block
    ) {
        blocks.push_back(
            make_block(
                generator,
                model_dimension,
                hidden_dimension,
                head_count
            )
        );
    }

    edge_llm::Linear output_projection(
        random_tensor(
            {vocabulary_size, model_dimension},
            generator,
            0.05F
        )
    );

    return edge_llm::TransformerModel(
        std::move(embedding),
        std::move(blocks),
        ones(model_dimension),
        std::move(output_projection)
    );
}

ErrorMetrics calculate_error(
    const edge_llm::Tensor& actual,
    const edge_llm::Tensor& reference
) {
    if (actual.shape() != reference.shape()) {
        throw std::invalid_argument(
            "Error comparison requires matching shapes"
        );
    }

    double total_error = 0.0;
    double maximum_error = 0.0;

    for (
        std::size_t index = 0;
        index < actual.size();
        ++index
    ) {
        const double error =
            std::fabs(
                static_cast<double>(actual.at(index)) -
                static_cast<double>(reference.at(index))
            );

        total_error += error;
        maximum_error =
            std::max(maximum_error, error);
    }

    return ErrorMetrics{
        maximum_error,
        total_error /
            static_cast<double>(actual.size())
    };
}

void print_timing(
    std::string_view prefix,
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

TimingStats measure_cached_decode(
    const edge_llm::TransformerModel& model,
    const std::vector<std::size_t>& prompt,
    std::size_t decode_steps,
    std::size_t warmup_iterations,
    std::size_t measured_iterations
) {
    std::vector<double> durations;
    durations.reserve(measured_iterations);

    double checksum = 0.0;

    const std::size_t total_iterations =
        warmup_iterations +
        measured_iterations;

    for (
        std::size_t iteration = 0;
        iteration < total_iterations;
        ++iteration
    ) {
        edge_llm::TransformerModel::LayerCaches caches =
            model.create_caches(
                prompt.size() + decode_steps
            );

        edge_llm::Tensor logits =
            model.prefill(prompt, caches);

        const bool measured =
            iteration >= warmup_iterations;

        const auto start = Clock::now();

        for (
            std::size_t step = 0;
            step < decode_steps;
            ++step
        ) {
            const std::size_t token_id =
                (
                    step * 17 +
                    iteration
                ) % model.vocabulary_size();

            logits =
                model.decode(
                    token_id,
                    caches
                );
        }

        const auto end = Clock::now();

        checksum += logits.at(
            logits.size() - 1
        );

        if (measured) {
            durations.push_back(
                std::chrono::duration<double, std::milli>(
                    end - start
                ).count()
            );
        }
    }

    return summarize(
        std::move(durations),
        checksum
    );
}

void print_environment() {
#if defined(__APPLE__)
    std::cout << "platform=macos\n";
#elif defined(__linux__)
    std::cout << "platform=linux\n";
#elif defined(_WIN32)
    std::cout << "platform=windows\n";
#else
    std::cout << "platform=unknown\n";
#endif

#if defined(__aarch64__) || defined(__arm64__)
    std::cout << "architecture=arm64\n";
#elif defined(__x86_64__) || defined(_M_X64)
    std::cout << "architecture=x86_64\n";
#else
    std::cout << "architecture=unknown\n";
#endif

#if defined(__clang__)
    std::cout
        << "compiler=clang-"
        << __clang_major__ << '.'
        << __clang_minor__ << '.'
        << __clang_patchlevel__ << '\n';
#elif defined(__GNUC__)
    std::cout
        << "compiler=gcc-"
        << __GNUC__ << '.'
        << __GNUC_MINOR__ << '.'
        << __GNUC_PATCHLEVEL__ << '\n';
#else
    std::cout << "compiler=unknown\n";
#endif

#ifdef NDEBUG
    std::cout << "build_type=Release\n";
#else
    std::cout << "build_type=Debug\n";
#endif
}

}  // namespace

int main() {
    try {
        constexpr std::size_t random_seed = 42;

        constexpr std::size_t linear_rows = 32;
        constexpr std::size_t linear_input_features = 256;
        constexpr std::size_t linear_output_features = 256;
        constexpr std::size_t linear_warmups = 5;
        constexpr std::size_t linear_iterations = 25;

        std::mt19937 linear_generator(random_seed);

        const edge_llm::Tensor linear_weight =
            random_tensor(
                {
                    linear_output_features,
                    linear_input_features
                },
                linear_generator,
                0.25F
            );

        const edge_llm::Tensor linear_input =
            random_tensor(
                {
                    linear_rows,
                    linear_input_features
                },
                linear_generator,
                1.0F
            );

        const edge_llm::Linear fp32_linear(
            linear_weight
        );

        const edge_llm::QuantizedLinear int8_linear(
            linear_weight
        );

        const edge_llm::Tensor fp32_reference =
            fp32_linear.forward(linear_input);

        const edge_llm::Tensor int8_reference =
            int8_linear.forward(linear_input);

        const ErrorMetrics linear_error =
            calculate_error(
                int8_reference,
                fp32_reference
            );

        const TimingStats fp32_timing =
            measure(
                [&fp32_linear, &linear_input] {
                    return fp32_linear.forward(
                        linear_input
                    );
                },
                [](const edge_llm::Tensor& output) {
                    return static_cast<double>(
                        output.at(output.size() - 1)
                    );
                },
                linear_warmups,
                linear_iterations
            );

        const TimingStats int8_timing =
            measure(
                [&int8_linear, &linear_input] {
                    return int8_linear.forward(
                        linear_input
                    );
                },
                [](const edge_llm::Tensor& output) {
                    return static_cast<double>(
                        output.at(output.size() - 1)
                    );
                },
                linear_warmups,
                linear_iterations
            );

        constexpr std::size_t vocabulary_size = 128;
        constexpr std::size_t model_dimension = 32;
        constexpr std::size_t hidden_dimension = 64;
        constexpr std::size_t head_count = 4;
        constexpr std::size_t block_count = 2;
        constexpr std::size_t prompt_length = 16;
        constexpr std::size_t generated_tokens = 8;
        constexpr std::size_t generation_warmups = 1;
        constexpr std::size_t generation_iterations = 5;

        std::mt19937 model_generator(random_seed);

        const edge_llm::TransformerModel model =
            make_model(
                model_generator,
                vocabulary_size,
                model_dimension,
                hidden_dimension,
                head_count,
                block_count
            );

        std::vector<std::size_t> prompt;
        prompt.reserve(prompt_length);

        for (
            std::size_t position = 0;
            position < prompt_length;
            ++position
        ) {
            prompt.push_back(
                (
                    position * 17 + 3
                ) % vocabulary_size
            );
        }

        const std::vector<std::size_t> uncached_reference =
            model.generate(
                prompt,
                generated_tokens
            );

        const std::vector<std::size_t> cached_reference =
            model.generate_cached(
                prompt,
                generated_tokens
            );

        if (uncached_reference != cached_reference) {
            throw std::runtime_error(
                "Cached and uncached generation produced "
                "different token sequences"
            );
        }

        const TimingStats uncached_generation_timing =
            measure(
                [&model, &prompt] {
                    return model.generate(
                        prompt,
                        generated_tokens
                    );
                },
                [](const std::vector<std::size_t>& tokens) {
                    return static_cast<double>(
                        tokens.back()
                    );
                },
                generation_warmups,
                generation_iterations
            );

        const TimingStats cached_generation_timing =
            measure(
                [&model, &prompt] {
                    return model.generate_cached(
                        prompt,
                        generated_tokens
                    );
                },
                [](const std::vector<std::size_t>& tokens) {
                    return static_cast<double>(
                        tokens.back()
                    );
                },
                generation_warmups,
                generation_iterations
            );

        const TimingStats cached_decode_timing =
            measure_cached_decode(
                model,
                prompt,
                generated_tokens,
                generation_warmups,
                generation_iterations
            );

        const double fp32_rows_per_second =
            static_cast<double>(linear_rows) /
            (fp32_timing.mean_ms / 1000.0);

        const double int8_rows_per_second =
            static_cast<double>(linear_rows) /
            (int8_timing.mean_ms / 1000.0);

        const double uncached_tokens_per_second =
            static_cast<double>(generated_tokens) /
            (
                uncached_generation_timing.mean_ms /
                1000.0
            );

        const double cached_tokens_per_second =
            static_cast<double>(generated_tokens) /
            (
                cached_generation_timing.mean_ms /
                1000.0
            );

        const std::size_t cache_capacity =
            prompt_length + generated_tokens - 1;

        const std::size_t kv_cache_payload_bytes =
            block_count *
            cache_capacity *
            model_dimension *
            2 *
            sizeof(float);

        std::cout
            << std::fixed
            << std::setprecision(4);

        std::cout
            << "edge_llm_benchmark_version=1\n";

        print_environment();

        std::cout
            << "random_seed=" << random_seed << "\n\n";

        std::cout << "[linear]\n";
        std::cout
            << "rows=" << linear_rows << '\n'
            << "input_features="
            << linear_input_features << '\n'
            << "output_features="
            << linear_output_features << '\n'
            << "warmup_iterations="
            << linear_warmups << '\n'
            << "measured_iterations="
            << linear_iterations << '\n'
            << "fp32_weight_bytes="
            << int8_linear.float_weight_bytes() << '\n'
            << "int8_weight_and_scale_bytes="
            << int8_linear.quantized_weight_bytes() << '\n'
            << "weight_compression_ratio="
            << int8_linear.compression_ratio() << '\n';

        print_timing(
            "fp32_linear",
            fp32_timing
        );

        print_timing(
            "int8_linear",
            int8_timing
        );

        std::cout
            << "fp32_rows_per_second="
            << fp32_rows_per_second << '\n'
            << "int8_rows_per_second="
            << int8_rows_per_second << '\n'
            << "int8_over_fp32_speedup="
            << fp32_timing.mean_ms /
                int8_timing.mean_ms << '\n'
            << "maximum_absolute_error="
            << linear_error.maximum_absolute_error << '\n'
            << "mean_absolute_error="
            << linear_error.mean_absolute_error << "\n\n";

        std::cout << "[generation]\n";
        std::cout
            << "vocabulary_size="
            << vocabulary_size << '\n'
            << "model_dimension="
            << model_dimension << '\n'
            << "hidden_dimension="
            << hidden_dimension << '\n'
            << "head_count="
            << head_count << '\n'
            << "block_count="
            << block_count << '\n'
            << "prompt_tokens="
            << prompt_length << '\n'
            << "generated_tokens="
            << generated_tokens << '\n'
            << "warmup_iterations="
            << generation_warmups << '\n'
            << "measured_iterations="
            << generation_iterations << '\n'
            << "kv_cache_payload_bytes="
            << kv_cache_payload_bytes << '\n'
            << "generated_sequences_match=true\n";

        print_timing(
            "uncached_generation",
            uncached_generation_timing
        );

        print_timing(
            "cached_generation",
            cached_generation_timing
        );

        std::cout
            << "uncached_tokens_per_second="
            << uncached_tokens_per_second << '\n'
            << "cached_tokens_per_second="
            << cached_tokens_per_second << '\n'
            << "cached_over_uncached_speedup="
            << uncached_generation_timing.mean_ms /
                cached_generation_timing.mean_ms << '\n'
            << "cached_decode_mean_ms_per_token="
            << cached_decode_timing.mean_ms /
                static_cast<double>(generated_tokens)
            << '\n'
            << "benchmark_checksum="
            << (
                fp32_timing.checksum +
                int8_timing.checksum +
                uncached_generation_timing.checksum +
                cached_generation_timing.checksum +
                cached_decode_timing.checksum
            )
            << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr
            << "Benchmark failed: "
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}
