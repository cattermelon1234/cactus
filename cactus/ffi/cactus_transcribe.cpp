#include "cactus_ffi.h"
#include "cactus_utils.h"

using namespace cactus::ffi;

extern "C" {

int cactus_transcribe(
    cactus_model_t model,
    const char* audio_file_path,
    const char* prompt,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    cactus_token_callback callback,
    void* user_data,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    (void)model;
    (void)audio_file_path;
    (void)prompt;
    (void)options_json;
    (void)callback;
    (void)user_data;
    (void)pcm_buffer;
    (void)pcm_buffer_size;

    CACTUS_LOG_ERROR("transcribe", "No standalone ASR model available; transcription is not supported");
    handle_error_response("No standalone ASR model available; transcription is not supported", response_buffer, buffer_size);
    return -1;
}

int cactus_detect_language(
    cactus_model_t model,
    const char* audio_file_path,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const uint8_t* pcm_buffer,
    size_t pcm_buffer_size
) {
    (void)model;
    (void)audio_file_path;
    (void)options_json;
    (void)pcm_buffer;
    (void)pcm_buffer_size;

    CACTUS_LOG_ERROR("detect_language", "No standalone ASR model available; language detection is not supported");
    handle_error_response("No standalone ASR model available; language detection is not supported", response_buffer, buffer_size);
    return -1;
}

} // extern "C"
