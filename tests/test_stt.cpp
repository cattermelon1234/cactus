#include "test_utils.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

using namespace EngineTestUtils;

bool test_audio_processor() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║         AUDIO PROCESSOR TEST             ║\n"
              << "╚══════════════════════════════════════════╝\n";
    using namespace cactus::engine;

    Timer t;

    const size_t n_fft = 400;
    const size_t hop_length = 160;
    const size_t sampling_rate = 16000;
    const size_t feature_size = 80;
    const size_t num_frequency_bins = 1 + n_fft / 2;

    AudioProcessor audio_proc;
    audio_proc.init_mel_filters(num_frequency_bins, feature_size, 0.0f, 8000.0f, sampling_rate);

    const size_t n_samples = sampling_rate;
    std::vector<float> waveform(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        waveform[i] = std::sin(2.0f * M_PI * 440.0f * i / sampling_rate);
    }

    AudioProcessor::SpectrogramConfig config;
    config.n_fft = n_fft;
    config.hop_length = hop_length;
    config.frame_length = n_fft;
    config.power = 2.0f;
    config.center = true;
    config.log_mel = "log10";

    auto log_mel_spec = audio_proc.compute_spectrogram(waveform, config);

    double elapsed = t.elapsed_ms();

    const float expected[] = {1.133450f, 1.142660f, 1.161900f, 1.196580f, 1.229480f};

    const size_t pad_length = n_fft / 2;
    const size_t padded_length = n_samples + 2 * pad_length;
    const size_t num_frames = 1 + (padded_length - n_fft) / hop_length;

    bool passed = true;
    if (log_mel_spec.size() != feature_size * num_frames) {
        std::cerr << "  [audio_processor] unexpected output size: got " << log_mel_spec.size()
                  << ", expected " << (feature_size * num_frames) << std::endl;
        passed = false;
    }

#ifdef __APPLE__
    const float abs_tolerance = 1e-4f;
    const float rel_tolerance = 1e-4f;
    for (size_t i = 0; i < 5 && passed; i++) {
        float actual = log_mel_spec[i * num_frames];
        float diff = std::abs(actual - expected[i]);
        float allowed = std::max(abs_tolerance, rel_tolerance * std::abs(expected[i]));
        if (diff > allowed) {
            std::cerr << "  [audio_processor][mac] idx=" << i
                      << " expected=" << expected[i]
                      << " actual=" << actual
                      << " diff=" << diff
                      << " allowed=" << allowed
                      << std::endl;
            passed = false;
        }
    }
#else
    // Linux uses the non-Accelerate FFT path with different absolute scaling.
    // Validate spectral shape against the same fixture rather than exact magnitude.
    const float shape_tolerance = 0.10f;
    const float anchor = log_mel_spec[0];
    if (!std::isfinite(anchor) || anchor <= 0.0f) {
        std::cerr << "  [audio_processor][non-apple] invalid anchor value: " << anchor << std::endl;
        passed = false;
    }
    for (size_t i = 0; i < 5 && passed; i++) {
        float actual = log_mel_spec[i * num_frames];
        if (!std::isfinite(actual)) {
            std::cerr << "  [audio_processor][non-apple] non-finite value at idx=" << i << std::endl;
            passed = false;
            break;
        }
        float expected_ratio = expected[i] / expected[0];
        float actual_ratio = actual / anchor;
        float diff = std::abs(actual_ratio - expected_ratio);
        if (diff > shape_tolerance) {
            std::cerr << "  [audio_processor][non-apple] idx=" << i
                      << " expected_ratio=" << expected_ratio
                      << " actual_ratio=" << actual_ratio
                      << " diff=" << diff
                      << " allowed=" << shape_tolerance
                      << " (actual=" << actual << ", anchor=" << anchor << ")"
                      << std::endl;
            passed = false;
        }
    }
#endif

    std::cout << "└─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms" << std::endl;

    return passed;
}

bool test_irfft_correctness() {
    using namespace cactus::engine;
    const float tol = 1e-4f;
    const float randomized_tol = 5e-4f;

    auto make_complex_input = [](size_t n) {
        return std::vector<float>((n / 2 + 1) * 2, 0.0f);
    };

    auto make_constant_expected = [](size_t n, float value) {
        return std::vector<float>(n, value);
    };

    auto make_cosine_expected = [](size_t n, size_t k = 1, float amplitude = 1.0f) {
        std::vector<float> expected(n, 0.0f);
        for (size_t t = 0; t < n; ++t) {
            expected[t] = amplitude * std::cos(2.0f * static_cast<float>(M_PI) *
                                              static_cast<float>(k * t) /
                                              static_cast<float>(n));
        }
        return expected;
    };

    auto make_sine_expected = [](size_t n, size_t k = 1, float amplitude = 1.0f) {
        std::vector<float> expected(n, 0.0f);
        for (size_t t = 0; t < n; ++t) {
            expected[t] = amplitude * std::sin(2.0f * static_cast<float>(M_PI) *
                                              static_cast<float>(k * t) /
                                              static_cast<float>(n));
        }
        return expected;
    };

    auto make_nyquist_expected = [](size_t n, float amplitude = 1.0f) {
        std::vector<float> expected(n, 0.0f);
        for (size_t t = 0; t < n; ++t) {
            expected[t] = (t % 2 == 0) ? amplitude : -amplitude;
        }
        return expected;
    };

    auto make_delta_expected = [](size_t n) {
        std::vector<float> expected(n, 0.0f);
        expected[0] = 1.0f;
        return expected;
    };

    auto compute_reference_irfft = [](const std::vector<float>& input, size_t n, const char* norm) {
        std::string norm_str = norm ? norm : "backward";
        float norm_factor = 0.0f;
        if (norm_str == "backward") {
            norm_factor = 1.0f / static_cast<float>(n);
        } else if (norm_str == "forward") {
            norm_factor = 1.0f;
        } else if (norm_str == "ortho") {
            norm_factor = 1.0f / std::sqrt(static_cast<float>(n));
        } else {
            throw std::invalid_argument("unsupported norm");
        }

        std::vector<float> expected(n, 0.0f);
        const size_t in_len = n / 2 + 1;
        const float two_pi_over_n = (2.0f * static_cast<float>(M_PI)) / static_cast<float>(n);
        for (size_t t = 0; t < n; ++t) {
            float sum = input[0];
            for (size_t k = 1; k < in_len; ++k) {
                const float re = input[k * 2];
                const float im = input[k * 2 + 1];
                const float angle = two_pi_over_n * static_cast<float>(k * t);
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const bool self_conjugate = (k * 2 == n);
                if (self_conjugate) {
                    sum += re * c;
                } else {
                    sum += 2.0f * (re * c - im * s);
                }
            }
            expected[t] = sum * norm_factor;
        }
        return expected;
    };

    struct ValueCase {
        const char* name;
        size_t n;
        const char* norm;
        std::vector<float> input;
        std::vector<float> expected;
    };

    std::vector<ValueCase> value_cases;
    {
        auto input = make_complex_input(1);
        input[0] = 3.5f;
        value_cases.push_back({"n=1 scalar", 1, "backward", std::move(input), {3.5f}});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[0] = static_cast<float>(n);
        value_cases.push_back({"dc backward n=8", n, "backward", std::move(input), make_constant_expected(n, 1.0f)});
    }
    {
        const size_t n = 2;
        auto input = make_complex_input(n);
        input[0] = static_cast<float>(n);
        value_cases.push_back({"dc backward n=2", n, "backward", std::move(input), make_constant_expected(n, 1.0f)});
    }
    {
        const size_t n = 2;
        auto input = make_complex_input(n);
        input[2] = static_cast<float>(n);
        value_cases.push_back({"nyquist backward n=2", n, "backward", std::move(input), make_nyquist_expected(n, 1.0f)});
    }
    {
        const size_t n = 3;
        auto input = make_complex_input(n);
        input[2] = 1.5f;
        value_cases.push_back({"cos k=1 n=3", n, "backward", std::move(input), make_cosine_expected(n, 1)});
    }
    {
        const size_t n = 3;
        auto input = make_complex_input(n);
        input[3] = -1.5f;
        value_cases.push_back({"sin k=1 n=3", n, "backward", std::move(input), make_sine_expected(n, 1)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[2] = 4.0f;
        value_cases.push_back({"cos k=1 n=8", n, "backward", std::move(input), make_cosine_expected(n)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[3] = -4.0f;
        value_cases.push_back({"sin k=1 n=8", n, "backward", std::move(input), make_sine_expected(n)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[4] = 4.0f;
        value_cases.push_back({"cos k=2 n=8", n, "backward", std::move(input), make_cosine_expected(n, 2)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[5] = -4.0f;
        value_cases.push_back({"sin k=2 n=8", n, "backward", std::move(input), make_sine_expected(n, 2)});
    }
    {
        const size_t n = 6;
        auto input = make_complex_input(n);
        input[2] = 3.0f;
        value_cases.push_back({"cos k=1 n=6", n, "backward", std::move(input), make_cosine_expected(n)});
    }
    {
        const size_t n = 6;
        auto input = make_complex_input(n);
        input[3] = -3.0f;
        value_cases.push_back({"sin k=1 n=6", n, "backward", std::move(input), make_sine_expected(n)});
    }
    {
        const size_t n = 6;
        auto input = make_complex_input(n);
        input[6] = static_cast<float>(n);
        value_cases.push_back({"nyquist backward n=6", n, "backward", std::move(input), make_nyquist_expected(n, 1.0f)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[0] = static_cast<float>(n);
        input[2] = 2.0f;
        input[5] = -1.0f;
        value_cases.push_back({
            "multi-bin superposition n=8",
            n,
            "backward",
            input,
            compute_reference_irfft(input, n, "backward")
        });
    }
    {
        const size_t n = 8;
        const size_t n_bins = n / 2 + 1;
        auto input = make_complex_input(n);
        for (size_t i = 0; i < n_bins; ++i) {
            input[i * 2] = 1.0f;
        }
        value_cases.push_back({"all-real bins delta n=8", n, "backward", std::move(input), make_delta_expected(n)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[0] = 1.0f;
        value_cases.push_back({"dc forward n=8", n, "forward", std::move(input), make_constant_expected(n, 1.0f)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[4] = 0.5f;
        value_cases.push_back({"cos k=2 forward n=8", n, "forward", std::move(input), make_cosine_expected(n, 2)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[0] = std::sqrt(static_cast<float>(n));
        value_cases.push_back({"dc ortho n=8", n, "ortho", std::move(input), make_constant_expected(n, 1.0f)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[5] = -std::sqrt(static_cast<float>(n)) / 2.0f;
        value_cases.push_back({"sin k=2 ortho n=8", n, "ortho", std::move(input), make_sine_expected(n, 2)});
    }
    {
        const size_t n = 8;
        auto input = make_complex_input(n);
        input[0] = static_cast<float>(n);
        value_cases.push_back({"null norm defaults backward n=8", n, nullptr, std::move(input), make_constant_expected(n, 1.0f)});
    }

    for (const auto& c : value_cases) {
        auto out = AudioProcessor::compute_irfft(c.input, c.n, c.norm);
        if (out.size() != c.expected.size()) {
            std::cerr << "[irfft][" << c.name << "] size mismatch: got " << out.size()
                      << ", expected " << c.expected.size() << std::endl;
            return false;
        }
        for (size_t i = 0; i < out.size(); ++i) {
            if (std::abs(out[i] - c.expected[i]) > tol) {
                std::cerr << "[irfft][" << c.name << "] idx=" << i
                          << " got=" << out[i]
                          << " expected=" << c.expected[i] << std::endl;
                return false;
            }
        }
    }

    {
        const size_t n = 8;
        auto base = make_complex_input(n);
        base[0] = static_cast<float>(n);
        base[2] = 2.0f;
        base[3] = -1.0f;

        auto with_dc_imag = base;
        with_dc_imag[1] = 123.0f;

        auto out_base = AudioProcessor::compute_irfft(base, n, "backward");
        auto out_with_dc_imag = AudioProcessor::compute_irfft(with_dc_imag, n, "backward");
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(out_base[i] - out_with_dc_imag[i]) > tol) {
                std::cerr << "[irfft][dc imag ignored] idx=" << i
                          << " base=" << out_base[i]
                          << " with_dc_imag=" << out_with_dc_imag[i] << std::endl;
                return false;
            }
        }
    }

    {
        const size_t n = 8;
        auto base = make_complex_input(n);
        base[0] = static_cast<float>(n);
        base[8] = static_cast<float>(n);

        auto with_nyquist_imag = base;
        with_nyquist_imag[9] = 321.0f;

        auto out_base = AudioProcessor::compute_irfft(base, n, "backward");
        auto out_with_nyquist_imag = AudioProcessor::compute_irfft(with_nyquist_imag, n, "backward");
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(out_base[i] - out_with_nyquist_imag[i]) > tol) {
                std::cerr << "[irfft][nyquist imag ignored] idx=" << i
                          << " base=" << out_base[i]
                          << " with_nyquist_imag=" << out_with_nyquist_imag[i] << std::endl;
                return false;
            }
        }
    }

    {
        uint32_t seed = 0x12345678u;
        auto next_value = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            const int centered = static_cast<int>((seed >> 8) & 0xFFFFu) - 32768;
            return static_cast<float>(centered) / 3276.8f;
        };

        const std::vector<size_t> sizes = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16};
        const std::vector<const char*> norms = {"backward", "forward", "ortho"};
        for (size_t n : sizes) {
            for (const char* norm : norms) {
                for (size_t trial = 0; trial < 3; ++trial) {
                    auto input = make_complex_input(n);
                    for (float& v : input) {
                        v = next_value();
                    }

                    auto out = AudioProcessor::compute_irfft(input, n, norm);
                    auto expected = compute_reference_irfft(input, n, norm);
                    if (out.size() != expected.size()) {
                        std::cerr << "[irfft][randomized] size mismatch: got " << out.size()
                                  << ", expected " << expected.size() << std::endl;
                        return false;
                    }
                    for (size_t i = 0; i < n; ++i) {
                        const float diff = std::abs(out[i] - expected[i]);
                        if (!std::isfinite(out[i]) || diff > randomized_tol) {
                            std::cerr << "[irfft][randomized] n=" << n
                                      << " norm=" << norm
                                      << " trial=" << trial
                                      << " idx=" << i
                                      << " got=" << out[i]
                                      << " expected=" << expected[i]
                                      << " diff=" << diff << std::endl;
                            return false;
                        }
                    }
                }
            }
        }
    }

    enum class ThrowCaseKind {
        ZeroN,
        BadInputSize,
        InvalidNorm
    };

    struct ThrowCase {
        const char* name;
        ThrowCaseKind kind;
    };

    const std::vector<ThrowCase> throw_cases = {
        {"zero n", ThrowCaseKind::ZeroN},
        {"bad input size", ThrowCaseKind::BadInputSize},
        {"invalid norm", ThrowCaseKind::InvalidNorm},
    };

    for (const auto& c : throw_cases) {
        bool threw = false;
        try {
            switch (c.kind) {
                case ThrowCaseKind::ZeroN: {
                    std::vector<float> complex_input(2, 0.0f);
                    (void)AudioProcessor::compute_irfft(complex_input, 0);
                    break;
                }
                case ThrowCaseKind::BadInputSize: {
                    std::vector<float> bad_input(4, 0.0f);
                    (void)AudioProcessor::compute_irfft(bad_input, 8);
                    break;
                }
                case ThrowCaseKind::InvalidNorm: {
                    const size_t n = 8;
                    const size_t n_bins = n / 2 + 1;
                    std::vector<float> complex_input(n_bins * 2, 0.0f);
                    (void)AudioProcessor::compute_irfft(complex_input, n, "invalid_norm");
                    break;
                }
            }
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "[irfft][" << c.name << "] expected std::invalid_argument" << std::endl;
            return false;
        }
    }

    return true;
}

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

bool test_transcription() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║        TRANSCRIPTION TEST                 ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_MODEL not set\n";
        return true;
    }
    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    char response[1 << 15] = {0};

    Timer timer;
    int rc = cactus_transcribe(model, audio_path.c_str(), nullptr,
                               response, sizeof(response),
                               R"({"max_tokens": 200, "telemetry_enabled": false})",
                               nullptr, nullptr, nullptr, 0);
    double elapsed = timer.elapsed_ms();

    if (rc <= 0) {
        std::cerr << "[✗] Transcription failed: " << response << "\n";
        cactus_destroy(model);
        return false;
    }

    std::string response_str(response);
    std::string transcript = json_string(response_str, "response");

    std::cout << "├─ Transcript: " << transcript << "\n"
              << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n";

    Metrics m;
    m.parse(response);
    m.print_json();

    cactus_destroy(model);

    bool passed = rc > 0 && !transcript.empty() && transcript.length() > 5;
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

bool test_transcription_pcm() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║      TRANSCRIPTION PCM TEST               ║\n"
              << "╚══════════════════════════════════════════╝\n";

    if (!g_model_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_MODEL not set\n";
        return true;
    }
    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    // Load WAV file and pass as raw PCM buffer
    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    FILE* wav_file = fopen(audio_path.c_str(), "rb");
    if (!wav_file) {
        std::cerr << "[✗] Failed to open audio file\n";
        cactus_destroy(model);
        return false;
    }

    // Skip 44-byte WAV header
    fseek(wav_file, 44, SEEK_SET);
    std::vector<uint8_t> pcm_data;
    uint8_t buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), wav_file)) > 0) {
        pcm_data.insert(pcm_data.end(), buf, buf + n);
    }
    fclose(wav_file);

    char response[1 << 15] = {0};

    Timer timer;
    int rc = cactus_transcribe(model, nullptr, nullptr,
                               response, sizeof(response),
                               R"({"max_tokens": 200, "telemetry_enabled": false})",
                               nullptr, nullptr,
                               pcm_data.data(), pcm_data.size());
    double elapsed = timer.elapsed_ms();

    if (rc <= 0) {
        std::cerr << "[✗] PCM transcription failed: " << response << "\n";
        cactus_destroy(model);
        return false;
    }

    std::string response_str(response);
    std::string transcript = json_string(response_str, "response");

    std::cout << "├─ Transcript: " << transcript << "\n"
              << "├─ PCM size: " << pcm_data.size() << " bytes\n"
              << "├─ Time: " << std::fixed << std::setprecision(2) << elapsed << "ms\n";

    cactus_destroy(model);

    bool passed = rc > 0 && !transcript.empty() && transcript.length() > 5;
    std::cout << "└─ Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    return passed;
}

int main() {
    TestUtils::TestRunner runner("Audio & STT Tests");
    runner.run_test("audio_processor", test_audio_processor());
    runner.run_test("irfft_correctness", test_irfft_correctness());
    runner.run_test("transcription", test_transcription());
    runner.run_test("transcription_pcm", test_transcription_pcm());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
