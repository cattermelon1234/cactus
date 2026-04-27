#include "test_utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace TestUtils;

bool test_matmul_f16() {
    const size_t M = 4, K = 8, N = 4;
    std::vector<__fp16> a(M * K), b(N * K), c(M * N);
    fill_random_fp16(a, -0.5f, 0.5f);
    fill_random_fp16(b, -0.5f, 0.5f);
    cactus_matmul_f16(a.data(), b.data(), c.data(), M, K, N);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            float ref = 0.0f;
            for (size_t k = 0; k < K; k++)
                ref += static_cast<float>(a[i * K + k]) * static_cast<float>(b[j * K + k]);
            if (std::abs(static_cast<float>(c[i * N + j]) - ref) > 0.05f) return false;
        }
    return true;
}

bool test_matmul_int8_grouped() {
    const size_t M = 4, K = 128, N = 8, gs = 32;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> id(-50, 50);
    std::uniform_real_distribution<float> sd(0.001f, 0.05f);
    std::vector<int8_t> A(M * K), B(N * K);
    std::vector<float> As(M);
    std::vector<__fp16> Bs(N * (K / gs)), C(M * N), Cv(N);
    for (auto& v : A) v = static_cast<int8_t>(id(gen));
    for (auto& v : B) v = static_cast<int8_t>(id(gen));
    for (auto& v : As) v = sd(gen);
    for (auto& v : Bs) v = static_cast<__fp16>(sd(gen));

    cactus_matmul_int8(A.data(), As.data(), B.data(), Bs.data(), C.data(), M, K, N, gs);
    for (size_t i = 0; i < M * N; i++)
        if (!std::isfinite(static_cast<float>(C[i]))) return false;

    cactus_gemv_int8(A.data(), As[0], B.data(), Bs.data(), Cv.data(), K, N, gs);
    for (size_t j = 0; j < N; j++) {
        float tol = std::abs(static_cast<float>(C[j])) * 0.05f + 0.5f;
        if (std::abs(static_cast<float>(C[j]) - static_cast<float>(Cv[j])) > tol) return false;
    }
    return true;
}

bool test_matmul_int4() {
    const size_t M = 2, K = 128, N = 8, gs = 32;
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> id(-5, 5);
    std::uniform_real_distribution<float> sd(0.01f, 0.1f);
    std::vector<int8_t> A(M * K), Bf(N * K), Bp(N * K / 2);
    std::vector<float> As(M);
    std::vector<__fp16> Bs(N * (K / gs)), C(M * N);
    for (auto& v : A) v = static_cast<int8_t>(id(gen));
    for (auto& v : As) v = sd(gen);
    for (auto& v : Bs) v = static_cast<__fp16>(sd(gen));
    for (auto& v : Bf) v = static_cast<int8_t>(std::max(-7, std::min(7, id(gen))));
    for (size_t i = 0; i < N * K / 2; i++)
        Bp[i] = static_cast<int8_t>((Bf[i * 2 + 1] << 4) | (Bf[i * 2] & 0x0F));

    cactus_matmul_int4(A.data(), As.data(), Bp.data(), Bs.data(), C.data(), M, K, N, gs);
    for (size_t i = 0; i < M * N; i++)
        if (!std::isfinite(static_cast<float>(C[i]))) return false;
    return true;
}

bool run_benchmarks() {
    auto bench = [](const char* label, size_t M, size_t K, size_t N, auto fn) {
        fn();
        Timer t;
        for (int i = 0; i < 100; i++) fn();
        double ms = t.elapsed_ms() / 100.0;
        double gflops = (2.0 * M * K * N) / (ms * 1e6);
        std::cout << "  ⚡ " << std::left << std::setw(28) << label
                  << std::fixed << std::setprecision(3) << ms << "ms  "
                  << std::setprecision(1) << gflops << " GFLOPS\n";
    };

    {
        const size_t K = 1024, N = 1024;
        std::vector<__fp16> a(K), b(N * K), c(N);
        fill_random_fp16(a, -0.5f, 0.5f); fill_random_fp16(b, -0.5f, 0.5f);
        bench("matmul_f16 1x1024x1024", 1, K, N, [&]{ cactus_matmul_f16(a.data(), b.data(), c.data(), 1, K, N); });
    }
    {
        const size_t M = 1024, K = 1024, N = 1024;
        std::vector<__fp16> a(M * K), b(N * K), c(M * N);
        fill_random_fp16(a, -0.5f, 0.5f); fill_random_fp16(b, -0.5f, 0.5f);
        bench("matmul_f16 1024^3", M, K, N, [&]{ cactus_matmul_f16(a.data(), b.data(), c.data(), M, K, N); });
    }
    {
        const size_t K = 1024, N = 1024, gs = 32;
        std::vector<int8_t> A(K), B(N * K);
        std::vector<__fp16> Bs(N * (K / gs)), C(N);
        bench("matmul_int8 1x1024x1024", 1, K, N, [&]{ cactus_gemv_int8(A.data(), 0.01f, B.data(), Bs.data(), C.data(), K, N, gs); });
    }
    {
        const size_t M = 1024, K = 1024, N = 1024, gs = 32;
        std::vector<int8_t> A(M * K), B(N * K);
        std::vector<float> As(M, 0.01f);
        std::vector<__fp16> Bs(N * (K / gs), static_cast<__fp16>(0.01f)), C(M * N);
        bench("matmul_int8 1024^3", M, K, N, [&]{ cactus_matmul_int8(A.data(), As.data(), B.data(), Bs.data(), C.data(), M, K, N, gs); });
    }
    {
        const size_t K = 1024, N = 1024, gs = 32;
        std::vector<int8_t> A(K), Bp(N * K / 2);
        std::vector<__fp16> Bs(N * (K / gs)), C(N);
        bench("matmul_int4 1x1024x1024", 1, K, N, [&]{ cactus_gemv_int4(A.data(), 0.01f, Bp.data(), Bs.data(), C.data(), K, N, gs); });
    }
    {
        const size_t M = 1024, K = 1024, N = 1024, gs = 32;
        std::vector<int8_t> A(M * K), Bp(N * K / 2);
        std::vector<float> As(M, 0.01f);
        std::vector<__fp16> Bs(N * (K / gs), static_cast<__fp16>(0.01f)), C(M * N);
        bench("matmul_int4 1024^3", M, K, N, [&]{ cactus_matmul_int4(A.data(), As.data(), Bp.data(), Bs.data(), C.data(), M, K, N, gs); });
    }
    return true;
}

int main() {
    TestRunner runner("Matrix Multiplication");
    runner.run_test("matmul_f16", test_matmul_f16());
    runner.run_test("matmul_int8_grouped", test_matmul_int8_grouped());
    runner.run_test("matmul_int4", test_matmul_int4());
    runner.print_benchmarks_header();
    runner.run_bench("benchmarks", run_benchmarks());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
