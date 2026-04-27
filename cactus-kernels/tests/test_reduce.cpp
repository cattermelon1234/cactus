#include "test_utils.h"
#include <vector>
#include <cmath>

using namespace TestUtils;

bool test_sum_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    double result = cactus_sum_all_f16(data.data(), n);
    double expected = 0.0;
    for (size_t i = 0; i < n; i++) expected += static_cast<double>(static_cast<float>(data[i]));

    if (std::abs(result - expected) > 0.5) {
        std::cerr << "  sum_all: " << result << " vs " << expected << "\n";
        return false;
    }
    return true;
}

bool test_mean_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    double result = cactus_mean_all_f16(data.data(), n);
    double expected = 0.0;
    for (size_t i = 0; i < n; i++) expected += static_cast<double>(static_cast<float>(data[i]));
    expected /= n;

    if (std::abs(result - expected) > 0.01) {
        std::cerr << "  mean_all: " << result << " vs " << expected << "\n";
        return false;
    }
    return true;
}

bool test_variance_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    double result = cactus_variance_all_f16(data.data(), n);

    double mean = 0.0;
    for (size_t i = 0; i < n; i++) mean += static_cast<float>(data[i]);
    mean /= n;
    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = static_cast<float>(data[i]) - mean;
        var += d * d;
    }
    var /= n;

    if (std::abs(result - var) > 0.02) {
        std::cerr << "  variance_all: " << result << " vs " << var << "\n";
        return false;
    }
    return true;
}

bool test_min_max_all() {
    const size_t n = 256;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -10.0f, 10.0f);

    __fp16 result_min = cactus_min_all_f16(data.data(), n);
    __fp16 result_max = cactus_max_all_f16(data.data(), n);

    __fp16 expected_min = data[0], expected_max = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] < expected_min) expected_min = data[i];
        if (data[i] > expected_max) expected_max = data[i];
    }

    if (result_min != expected_min || result_max != expected_max) {
        std::cerr << "  min/max: got [" << static_cast<float>(result_min) << ", "
                  << static_cast<float>(result_max) << "] expected ["
                  << static_cast<float>(expected_min) << ", "
                  << static_cast<float>(expected_max) << "]\n";
        return false;
    }
    return true;
}

bool test_sum_axis() {
    const size_t outer = 4, axis = 8, inner = 16;
    std::vector<__fp16> input(outer * axis * inner), output(outer * inner);
    fill_random_fp16(input, -1.0f, 1.0f);

    cactus_sum_axis_f16(input.data(), output.data(), outer, axis, inner);

    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float ref = 0.0f;
            for (size_t a = 0; a < axis; a++) {
                ref += static_cast<float>(input[(o * axis + a) * inner + i]);
            }
            float actual = static_cast<float>(output[o * inner + i]);
            if (std::abs(actual - ref) > 0.1f) {
                std::cerr << "  sum_axis mismatch [" << o << "," << i << "]: "
                          << actual << " vs " << ref << "\n";
                return false;
            }
        }
    }
    return true;
}

bool run_benchmarks() {
    auto bench = [](const char* label, size_t n, auto fn) {
        fn();
        Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        double gb_s = (static_cast<double>(n) * sizeof(__fp16)) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << label
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gb_s << " GB/s\n";
    };

    const size_t n = 1024 * 1024;
    std::vector<__fp16> data(n);
    fill_random_fp16(data, -1.0f, 1.0f);

    volatile double sink_d = 0.0;
    volatile __fp16 sink_h = static_cast<__fp16>(0.0f);

    bench("sum_all 1M", n, [&]{ sink_d = cactus_sum_all_f16(data.data(), n); });
    bench("mean_all 1M", n, [&]{ sink_d = cactus_mean_all_f16(data.data(), n); });
    bench("variance_all 1M", n, [&]{ sink_d = cactus_variance_all_f16(data.data(), n); });
    bench("min_all 1M", n, [&]{ sink_h = cactus_min_all_f16(data.data(), n); });
    bench("max_all 1M", n, [&]{ sink_h = cactus_max_all_f16(data.data(), n); });

    (void)sink_d; (void)sink_h;
    return true;
}

int main() {
    TestRunner runner("Reduction Operations");
    runner.run_test("sum_all", test_sum_all());
    runner.run_test("mean_all", test_mean_all());
    runner.run_test("variance_all", test_variance_all());
    runner.run_test("min_max_all", test_min_max_all());
    runner.run_test("sum_axis", test_sum_axis());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
