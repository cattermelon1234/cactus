#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../../libs/audio/wav.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <string>

using namespace cactus::engine;
using namespace cactus::ffi;
using cactus::audio::WHISPER_SAMPLE_RATE;
using cactus::audio::apply_preemphasis;
using cactus::audio::get_parakeet_spectrogram_config;
using cactus::audio::get_whisper_spectrogram_config;
using cactus::audio::init_whisper_mel_filters;
using cactus::audio::normalize_parakeet_log_mel;
using cactus::audio::normalize_whisper_mel;
using cactus::audio::preprocess_audio_for_gemma4;
using cactus::audio::trim_mel_frames;

static std::vector<float> compute_mel_from_wav(const std::string& wav_path, bool is_parakeet, size_t mel_bins) {
    AudioFP32 audio = load_wav(wav_path);
    std::vector<float> waveform_16k = resample_to_16k_fp32(audio.samples, audio.sample_rate);

    auto cfg = is_parakeet ? get_parakeet_spectrogram_config() : get_whisper_spectrogram_config();
    const size_t num_mel_filters = std::max<size_t>(1, mel_bins);

    AudioProcessor ap;
    if (is_parakeet) {
        ap.init_mel_filters(cfg.n_fft / 2 + 1, num_mel_filters, 0.0f, 8000.0f, WHISPER_SAMPLE_RATE);
    } else {
        init_whisper_mel_filters(ap, cfg, num_mel_filters);
    }
    const size_t waveform_samples = waveform_16k.size();
    if (is_parakeet) {
        apply_preemphasis(waveform_16k, 0.97f);
    }
    std::vector<float> mel = ap.compute_spectrogram(waveform_16k, cfg);

    if (mel.empty()) return mel;
    if (is_parakeet) {
        normalize_parakeet_log_mel(mel, num_mel_filters);
        size_t valid_frames = waveform_samples / cfg.hop_length;
        if (valid_frames == 0) valid_frames = 1;
        trim_mel_frames(mel, num_mel_filters, valid_frames);
        return mel;
    }

    const bool is_v3 = num_mel_filters > 80;
    return normalize_whisper_mel(mel, num_mel_filters, is_v3);
}

extern "C" {

int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim,
    bool normalize
) {
    if (!model || !text || !embeddings_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("embed", "Invalid parameters for text embedding");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();

        std::vector<uint32_t> tokens = tokenizer->encode(text);
        if (tokens.empty()) {
            CACTUS_LOG_ERROR("embed", "Tokenization produced empty result");
            return -1;
        }

        std::vector<float> embeddings = handle->model->get_embeddings(tokens, true, normalize);
        if (embeddings.size() * sizeof(float) > buffer_size) {
            CACTUS_LOG_ERROR("embed", "Buffer too small: need " << embeddings.size() * sizeof(float) << " bytes, got " << buffer_size);
            return -2;
        }

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) *embedding_dim = embeddings.size();

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("embed", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during embedding";
        CACTUS_LOG_ERROR("embed", last_error_message);
        return -1;
    }
}

int cactus_image_embed(
    cactus_model_t model,
    const char* image_path,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model || !image_path || !embeddings_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("image_embed", "Invalid parameters for image embedding");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);

        CACTUS_LOG_DEBUG("image_embed", "Processing image: " << image_path);
        std::vector<float> embeddings = handle->model->get_image_embeddings(image_path);
        if (embeddings.empty()) {
            CACTUS_LOG_ERROR("image_embed", "Image embedding returned empty result");
            return -1;
        }
        if (embeddings.size() * sizeof(float) > buffer_size) {
            CACTUS_LOG_ERROR("image_embed", "Buffer too small: need " << embeddings.size() * sizeof(float) << " bytes");
            return -2;
        }

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) *embedding_dim = embeddings.size();

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("image_embed", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during image embedding";
        CACTUS_LOG_ERROR("image_embed", last_error_message);
        return -1;
    }
}

int cactus_audio_embed(
    cactus_model_t model,
    const char* audio_path,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model || !audio_path || !embeddings_buffer || buffer_size == 0) {
        CACTUS_LOG_ERROR("audio_embed", "Invalid parameters for audio embedding");
        return -1;
    }

    try {
        auto* handle = static_cast<CactusModelHandle*>(model);

        CACTUS_LOG_DEBUG("audio_embed", "Processing audio: " << audio_path);
        const bool is_parakeet =
            handle->model->get_config().model_type == cactus::engine::Config::ModelType::PARAKEET ||
            handle->model->get_config().model_type == cactus::engine::Config::ModelType::PARAKEET_TDT;
        const bool is_gemma4 =
            handle->model->get_config().model_type == cactus::engine::Config::ModelType::GEMMA4;
        std::vector<float> mel_bins;
        if (is_gemma4) {
            AudioFP32 audio = load_wav(audio_path);
            std::vector<float> waveform_16k = resample_to_16k_fp32(audio.samples, audio.sample_rate);
            auto prepared = preprocess_audio_for_gemma4(std::move(waveform_16k), handle->model->get_config());
            mel_bins = std::move(prepared.features);
        } else {
            mel_bins = compute_mel_from_wav(audio_path, is_parakeet, handle->model->get_config().num_mel_bins);
        }
        if (mel_bins.empty()) {
            last_error_message = "Failed to compute mel spectrogram";
            CACTUS_LOG_ERROR("audio_embed", last_error_message << " for: " << audio_path);
            return -1;
        }

        std::vector<float> embeddings = handle->model->get_audio_embeddings(mel_bins);
        if (embeddings.empty()) {
            CACTUS_LOG_ERROR("audio_embed", "Audio embedding returned empty result");
            return -1;
        }
        if (embeddings.size() * sizeof(float) > buffer_size) {
            CACTUS_LOG_ERROR("audio_embed", "Buffer too small: need " << embeddings.size() * sizeof(float) << " bytes");
            return -2;
        }

        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) *embedding_dim = embeddings.size();

        return static_cast<int>(embeddings.size());

    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("audio_embed", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during audio embedding";
        CACTUS_LOG_ERROR("audio_embed", last_error_message);
        return -1;
    }
}

int cactus_preprocess_audio_features(
    const char* audio_path,
    const char* model_type,
    size_t mel_bins,
    float* features_buffer,
    size_t buffer_size,
    size_t* feature_count,
    size_t* out_mel_bins,
    size_t* out_frames
) {
    if (!audio_path || !model_type || !features_buffer || buffer_size == 0 || mel_bins == 0) {
        CACTUS_LOG_ERROR("audio_features", "Invalid parameters for audio feature preprocessing");
        return -1;
    }

    try {
        std::string type(model_type);
        std::transform(type.begin(), type.end(), type.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const bool is_parakeet = type.find("parakeet") != std::string::npos;
        const bool is_gemma4 = type.find("gemma4") != std::string::npos;

        std::vector<float> features;
        if (is_gemma4) {
            AudioFP32 audio = load_wav(audio_path);
            std::vector<float> waveform_16k = resample_to_16k_fp32(audio.samples, audio.sample_rate);
            cactus::engine::Config config;
            config.audio_input_feat_size = static_cast<uint32_t>(mel_bins);
            config.audio_fft_length = 512;
            auto prepared = preprocess_audio_for_gemma4(std::move(waveform_16k), config);
            features = std::move(prepared.features);
        } else {
            features = compute_mel_from_wav(audio_path, is_parakeet, mel_bins);
        }
        if (features.empty()) {
            last_error_message = "Failed to compute audio features";
            CACTUS_LOG_ERROR("audio_features", last_error_message << " for: " << audio_path);
            return -1;
        }
        if ((features.size() % mel_bins) != 0) {
            last_error_message = "Computed audio features have invalid mel/frame shape";
            CACTUS_LOG_ERROR("audio_features", last_error_message);
            return -1;
        }

        if (feature_count) *feature_count = features.size();
        if (out_mel_bins) *out_mel_bins = mel_bins;
        if (out_frames) *out_frames = features.size() / mel_bins;

        const size_t bytes = features.size() * sizeof(float);
        if (bytes > buffer_size) {
            CACTUS_LOG_ERROR("audio_features", "Buffer too small: need " << bytes << " bytes");
            return -2;
        }

        std::memcpy(features_buffer, features.data(), bytes);
        return static_cast<int>(features.size());
    } catch (const std::exception& e) {
        last_error_message = e.what();
        CACTUS_LOG_ERROR("audio_features", "Exception: " << e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during audio feature preprocessing";
        CACTUS_LOG_ERROR("audio_features", last_error_message);
        return -1;
    }
}

}
