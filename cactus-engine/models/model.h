#pragma once

#include "../src/engine.h"

namespace cactus {
namespace engine {

class QwenModel : public Model {
public:
    QwenModel();
    explicit QwenModel(const Config& config);
    ~QwenModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                           ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;
    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                     ComputeBackend backend) const override;
    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                   ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t pre_feedforward_layernorm_weight;
            size_t post_feedforward_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};

class Lfm2VlModel;

class Siglip2VisionModel : public Model {
    friend class Lfm2VlModel;

public:
    struct VisionEmbeddingResult {
        size_t combined_embeddings;
        std::vector<size_t> tile_embeddings;
    };

    Siglip2VisionModel();
    explicit Siglip2VisionModel(const Config& cfg);
    ~Siglip2VisionModel() override = default;

    virtual size_t forward_vision(const Siglip2Preprocessor::PreprocessedImage& preprocessed_image);
    virtual size_t forward_vision(CactusGraph* gb,
                                  const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
                                  ComputeBackend backend);
    std::vector<float> get_image_embedding(const std::string& image_path);
    Siglip2Preprocessor& get_preprocessor() { return preprocessor_; }
    const Siglip2Preprocessor& get_preprocessor() const { return preprocessor_; }

protected:
    VisionEmbeddingResult build_vision_embeddings(CactusGraph* gb,
                                                  const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
                                                  ComputeBackend backend);
    size_t build_vision_transformer_layer(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                                          ComputeBackend backend);
    size_t build_vision_attention(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                                  ComputeBackend backend);
    size_t build_vision_mlp(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                            ComputeBackend backend);
    void ensure_cpu_vision_weights_loaded(CactusGraph* gb);

    void load_weights_to_graph(CactusGraph* gb) override;
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                           ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;
    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                     ComputeBackend backend) const override;
    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                   ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    struct VisionWeightNodeIDs {
        size_t patch_embedding_weight;
        size_t patch_embedding_bias;
        size_t position_embedding;
        size_t post_layernorm_weight;
        size_t post_layernorm_bias;

        struct VisionLayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t attn_q_bias;
            size_t attn_k_bias;
            size_t attn_v_bias;
            size_t attn_output_bias;
            size_t layer_norm1_weight;
            size_t layer_norm1_bias;
            size_t layer_norm2_weight;
            size_t layer_norm2_bias;
            size_t mlp_fc1_weight;
            size_t mlp_fc1_bias;
            size_t mlp_fc2_weight;
            size_t mlp_fc2_bias;
        };

        std::vector<VisionLayerWeights> vision_layers;
    } vision_weight_nodes_;

    Siglip2Preprocessor preprocessor_;
    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
    bool cpu_vision_weights_loaded_ = false;
};

class LFM2Model : public Model {
    friend class Lfm2VlModel;

public:
    LFM2Model();
    explicit LFM2Model(const Config& config);
    ~LFM2Model() override = default;

    bool is_cache_empty() const;
    void update_kv_cache(CactusGraph* gb, size_t seq_len);
    bool init(const std::string& model_folder, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true) override;
    bool init(CactusGraph* external_graph, const std::string& model_folder, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true) override;

protected:
    using Model::forward;
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                           ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;
    size_t build_conv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                        ComputeBackend backend, bool use_cache);
    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                     ComputeBackend backend) const override;
    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                   ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    size_t forward(CactusGraph* gb, const std::vector<uint32_t>& tokens, ComputeBackend backend, bool use_cache = false);
    size_t forward(CactusGraph* gb, size_t input_embeddings, size_t seq_len, ComputeBackend backend, bool use_cache = false);
    void post_init() override;
    void post_execute_updates(CactusGraph* gb, size_t seq_len) override;
    void reset_cache() override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t conv_depthwise_weight;
            size_t conv_in_proj_weight;
            size_t conv_out_proj_weight;
            size_t input_layernorm_weight;
            size_t post_attention_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
        };

        enum class LayerType : uint8_t { ATTENTION, CONV };

        struct LayerEntry {
            LayerType type;
            LayerWeights weights;
        };

        std::vector<LayerEntry> layers;
    } weight_nodes_;

    ConvCache conv_cache_;
    std::vector<size_t> conv_cache_bx_nodes_;
    bool last_forward_used_cache_ = false;
};

class Lfm2VlModel : public Model {
public:
    Lfm2VlModel();
    explicit Lfm2VlModel(const Config& config);
    ~Lfm2VlModel() override = default;

    bool init(const std::string& model_folder, size_t context_size,
              const std::string& system_prompt = "", bool do_warmup = true) override;
    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    uint32_t decode(const std::vector<uint32_t>& tokens,
                    float temperature = -1.0f,
                    float top_p = -1.0f,
                    size_t top_k = 0,
                    const std::string& profile_file = "",
                    float* out_entropy = nullptr,
                    float min_p = 0.15f,
                    float repetition_penalty = 1.1f) override;
    void prefill(const std::vector<uint32_t>& tokens, size_t chunk_size = 256,
                 const std::string& profile_file = "") override;
    void prefill_with_images(const std::vector<uint32_t>& tokens,
                             const std::vector<std::string>& image_paths,
                             const std::string& profile_file = "") override;
    uint32_t decode_with_images(const std::vector<uint32_t>& tokens,
                                const std::vector<std::string>& image_paths,
                                float temperature = -1.0f,
                                float top_p = -1.0f,
                                size_t top_k = 0,
                                const std::string& profile_file = "",
                                float* out_entropy = nullptr,
                                float min_p = 0.15f,
                                float repetition_penalty = 1.1f) override;
    void reset_cache() override;
    std::vector<float> get_image_embeddings(const std::string& image_path) override;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override;
    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override;
    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct ProjectedTileFeature {
        size_t node_id;
        size_t token_count;
    };

    struct TextEmbeddingInput {
        size_t input_node;
        std::vector<uint32_t> tokens;
    };

    struct MergedEmbeddingResult {
        size_t node_id;
        size_t seq_len;
    };

    struct ForwardImageResult {
        size_t final_hidden_node;
        size_t seq_len;
    };

    std::vector<ProjectedTileFeature> get_image_features(
        CactusGraph* gb,
        const Siglip2Preprocessor::PreprocessedImage& preprocessed_image,
        ComputeBackend backend);
    ForwardImageResult forward_images(CactusGraph* gb,
                                      const std::vector<uint32_t>& tokens,
                                      const std::vector<std::string>& image_paths,
                                      ComputeBackend backend,
                                      bool use_cache);
    size_t build_multimodal_projector(CactusGraph* gb,
                                      size_t image_features,
                                      size_t tile_h,
                                      size_t tile_w,
                                      ComputeBackend backend);
    size_t pixel_unshuffle(CactusGraph* gb, size_t hidden_states, size_t height, size_t width, size_t channels);
    MergedEmbeddingResult merge_image_text_embeddings(
        CactusGraph* gb,
        const std::vector<uint32_t>& tokens,
        const std::vector<std::vector<ProjectedTileFeature>>& image_embedding_nodes,
        std::vector<TextEmbeddingInput>& text_embedding_inputs);

    Siglip2VisionModel vision_tower_;
    LFM2Model language_model_;
    Siglip2Preprocessor preprocessor_;

    struct ProjectorWeights {
        size_t layer_norm_weight;
        size_t layer_norm_bias;
        size_t linear_1_weight;
        size_t linear_1_bias;
        size_t linear_2_weight;
        size_t linear_2_bias;
    } projector_weights_;

    bool vision_weights_loaded_ = false;
    bool language_weights_loaded_ = false;
    bool image_prefill_completed_ = false;
    size_t last_token_count_ = 0;
};

class WhisperModel : public Model {
public:
    WhisperModel();
    explicit WhisperModel(const Config& config);
    ~WhisperModel() override = default;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Whisper: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("Whisper: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Whisper: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>&, bool = false) override {
        throw std::runtime_error("Whisper requires mel+token forward().");
    }

    size_t forward(const std::vector<float>& audio_features,
                   const std::vector<uint32_t>& tokens,
                   bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    uint32_t decode_with_audio(const std::vector<uint32_t>& tokens,
                               const std::vector<float>& audio_features,
                               float temperature = 0.0f,
                               float top_p = 0.0f,
                               size_t top_k = 0,
                               const std::string& profile_file = "",
                               float* out_entropy = nullptr,
                               float min_p = 0.15f,
                               float repetition_penalty = 1.1f,
                               float* out_token_time_start = nullptr,
                               float* out_token_time_end = nullptr) override;
    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features) override;
    void reset_cache() override;

private:
    void run_encoder(const std::vector<float>& audio_features);
    void reset_graph_side_cache_nodes();
    size_t run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only);

    size_t build_encoder_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                   ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    size_t build_decoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                        ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    size_t build_encoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                        ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    size_t build_encoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                             ComputeBackend backend);
    size_t build_decoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                             ComputeBackend backend) const;
    size_t build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                           ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    size_t build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                           ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    size_t build_conv1d(CactusGraph* gb, size_t input);

    struct WeightNodeIDs {
        size_t output_weight = 0;
        size_t decoder_norm_weight = 0;
        size_t decoder_norm_bias = 0;
        size_t decoder_position_embeddings_weight = 0;
        size_t encoder_position_embeddings = 0;
        size_t encoder_conv1_weight = 0;
        size_t encoder_conv1_bias = 0;
        size_t encoder_conv2_weight = 0;
        size_t encoder_conv2_bias = 0;
        size_t encoder_norm_weight = 0;
        size_t encoder_norm_bias = 0;

        struct LayerWeights {
            size_t decoder_encoder_attn_q_weight = 0;
            size_t decoder_encoder_attn_k_weight = 0;
            size_t decoder_encoder_attn_v_weight = 0;
            size_t decoder_encoder_attn_q_bias = 0;
            size_t decoder_encoder_attn_v_bias = 0;
            size_t decoder_encoder_attn_output_weight = 0;
            size_t decoder_encoder_attn_output_bias = 0;
            size_t decoder_post_encoder_layernorm_weight = 0;
            size_t decoder_post_encoder_layernorm_bias = 0;
            size_t decoder_ffn1_weight = 0;
            size_t decoder_ffn1_bias = 0;
            size_t decoder_ffn2_weight = 0;
            size_t decoder_ffn2_bias = 0;
            size_t decoder_post_ffn_layernorm_weight = 0;
            size_t decoder_post_ffn_layernorm_bias = 0;
            size_t decoder_self_attn_q_weight = 0;
            size_t decoder_self_attn_k_weight = 0;
            size_t decoder_self_attn_v_weight = 0;
            size_t decoder_self_attn_q_bias = 0;
            size_t decoder_self_attn_v_bias = 0;
            size_t decoder_self_attn_output_weight = 0;
            size_t decoder_self_attn_output_bias = 0;
            size_t decoder_post_attn_layernorm_weight = 0;
            size_t decoder_post_attn_layernorm_bias = 0;
            size_t encoder_ffn1_weight = 0;
            size_t encoder_ffn1_bias = 0;
            size_t encoder_ffn2_weight = 0;
            size_t encoder_ffn2_bias = 0;
            size_t encoder_post_ffn_layernorm_weight = 0;
            size_t encoder_post_ffn_layernorm_bias = 0;
            size_t encoder_self_attn_q_weight = 0;
            size_t encoder_self_attn_k_weight = 0;
            size_t encoder_self_attn_v_weight = 0;
            size_t encoder_self_attn_q_bias = 0;
            size_t encoder_self_attn_v_bias = 0;
            size_t encoder_self_attn_output_weight = 0;
            size_t encoder_self_attn_output_bias = 0;
            size_t encoder_post_attn_layernorm_weight = 0;
            size_t encoder_post_attn_layernorm_bias = 0;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;

    bool encoder_ready_ = false;
    size_t last_new_tokens_ = 0;
    size_t last_conv1_node_ = 0;
    size_t last_conv2_node_ = 0;
    size_t last_encoder_post_norm_node_ = 0;
    size_t last_enc_plus_pos_node_ = 0;
    size_t encoder_transformer_block_0 = 0;
    size_t encoder_pre_gelu = 0;
    size_t encoder_post_gelu = 0;
    std::vector<size_t> encoder_block_out_nodes_;
    size_t encoder_output_persistent_ = 0;
    std::vector<size_t> encoder_k_persistent_;
    std::vector<size_t> encoder_v_persistent_;
    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
    uint32_t enc_layers_ = 0;
    uint32_t dec_layers_ = 0;
    bool first_decode_step_ = true;
    std::unordered_map<uint32_t, float> suppress_bias_;
    std::unordered_map<uint32_t, float> suppress_bias_first_step_;
    std::vector<size_t> suppress_tokens_ = {
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
        90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350,
        1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
        6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
        13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075,
        21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
        36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362
    };
    std::vector<size_t> begin_suppress_tokens_ = {220, 50257};
};

class ParakeetTDTModel : public Model {
public:
    ParakeetTDTModel();
    explicit ParakeetTDTModel(const Config& config);
    ~ParakeetTDTModel() override = default;

    struct TDTToken { uint32_t id; float time_start; float time_end; };

    struct ChunkStreamState {
        bool initialized = false;
        uint32_t last_token = 0;
        std::vector<std::vector<__fp16>> h;
        std::vector<std::vector<__fp16>> c;
    };

    struct ChunkStreamResult {
        std::string text;
        std::string confirmed_text;
        std::string pending_text;
        size_t token_count = 0;
        size_t confirmed_token_count = 0;
        double raw_decoder_tps = 0.0;
        double raw_decoder_time_ms = 0.0;
        float start_sec = 0.0f;
        float confirmed_end_sec = 0.0f;
        float resume_end_sec = 0.0f;
        float end_sec = 0.0f;
    };

    ChunkStreamResult decode_chunk_stream(const std::vector<float>& audio_features,
                                          size_t replay_start_frame,
                                          size_t start_frame,
                                          size_t end_frame,
                                          ChunkStreamState& state);

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("ParakeetTDT: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("ParakeetTDT: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("ParakeetTDT: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>&, bool = false) override {
        throw std::runtime_error("ParakeetTDT requires audio feature forward().");
    }

    size_t forward(const std::vector<float>& audio_features,
                   const std::vector<uint32_t>& tokens,
                   bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    uint32_t decode_with_audio(const std::vector<uint32_t>& tokens,
                               const std::vector<float>& audio_features,
                               float temperature = 0.0f,
                               float top_p = 0.0f,
                               size_t top_k = 0,
                               const std::string& profile_file = "",
                               float* out_entropy = nullptr,
                               float min_p = 0.15f,
                               float repetition_penalty = 1.1f,
                               float* out_token_time_start = nullptr,
                               float* out_token_time_end = nullptr) override;
    std::vector<float> get_audio_embeddings(const std::vector<float>& audio_features) override;
    void reset_cache() override;

private:
    size_t build_encoder(CactusGraph* gb, const std::vector<float>& audio_features);
    size_t build_subsampling(CactusGraph* gb, const std::vector<float>& audio_features);
    size_t build_relative_position_embeddings(CactusGraph* gb, size_t seq_len);
    size_t build_self_attention(CactusGraph* gb, size_t hidden, size_t position_embeddings,
                                uint32_t layer_idx, ComputeBackend backend);
    size_t build_feed_forward(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                              bool second_ff, ComputeBackend backend);
    size_t build_convolution_module(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                    ComputeBackend backend);
    size_t build_encoder_block(CactusGraph* gb, size_t hidden, size_t position_embeddings,
                               uint32_t layer_idx, ComputeBackend backend);
    std::vector<TDTToken> decode_tdt_tokens_with_state(CactusGraph* gb,
                                                       size_t encoder_hidden_node,
                                                       size_t replay_start_frame,
                                                       size_t start_frame,
                                                       size_t end_frame,
                                                       ChunkStreamState* stream_state,
                                                       size_t* out_confirmed_count = nullptr,
                                                       double* out_raw_decoder_time_ms = nullptr) const;
    std::vector<TDTToken> greedy_decode_tdt_tokens(CactusGraph* gb, size_t encoder_hidden_node) const;

    struct WeightNodeIDs {
        size_t subsampling_conv0_weight = 0;
        size_t subsampling_conv0_bias = 0;
        size_t subsampling_depthwise1_weight = 0;
        size_t subsampling_depthwise1_bias = 0;
        size_t subsampling_pointwise1_weight = 0;
        size_t subsampling_pointwise1_bias = 0;
        size_t subsampling_depthwise2_weight = 0;
        size_t subsampling_depthwise2_bias = 0;
        size_t subsampling_pointwise2_weight = 0;
        size_t subsampling_pointwise2_bias = 0;
        size_t subsampling_linear_weight = 0;
        size_t subsampling_linear_bias = 0;

        struct LayerWeights {
            size_t ff1_linear1_weight = 0;
            size_t ff1_linear1_bias = 0;
            size_t ff1_linear2_weight = 0;
            size_t ff1_linear2_bias = 0;
            size_t ff2_linear1_weight = 0;
            size_t ff2_linear1_bias = 0;
            size_t ff2_linear2_weight = 0;
            size_t ff2_linear2_bias = 0;
            size_t self_attn_q_weight = 0;
            size_t self_attn_q_bias = 0;
            size_t self_attn_k_weight = 0;
            size_t self_attn_k_bias = 0;
            size_t self_attn_v_weight = 0;
            size_t self_attn_v_bias = 0;
            size_t self_attn_output_weight = 0;
            size_t self_attn_output_bias = 0;
            size_t self_attn_relative_k_weight = 0;
            size_t self_attn_bias_u = 0;
            size_t self_attn_bias_v = 0;
            size_t norm_ff1_weight = 0;
            size_t norm_ff1_bias = 0;
            size_t norm_self_attn_weight = 0;
            size_t norm_self_attn_bias = 0;
            size_t norm_conv_weight = 0;
            size_t norm_conv_bias = 0;
            size_t norm_ff2_weight = 0;
            size_t norm_ff2_bias = 0;
            size_t norm_out_weight = 0;
            size_t norm_out_bias = 0;
            size_t conv_pointwise1_weight = 0;
            size_t conv_pointwise1_bias = 0;
            size_t conv_depthwise_weight = 0;
            size_t conv_depthwise_bias = 0;
            size_t conv_pointwise2_weight = 0;
            size_t conv_pointwise2_bias = 0;
            size_t conv_batchnorm_weight = 0;
            size_t conv_batchnorm_bias = 0;
            size_t conv_batchnorm_running_mean = 0;
            size_t conv_batchnorm_running_var = 0;
        };

        struct PredictorLayerWeights {
            size_t weight_ih = 0;
            size_t weight_hh = 0;
            size_t bias = 0;
        };

        size_t predictor_embed = 0;
        std::vector<PredictorLayerWeights> predictor_layers;
        size_t joint_enc_weight = 0;
        size_t joint_enc_bias = 0;
        size_t joint_pred_weight = 0;
        size_t joint_pred_bias = 0;
        size_t joint_out_weight = 0;
        size_t joint_out_bias = 0;
        std::vector<LayerWeights> layers;
    } weight_nodes_;

    bool tdt_tokens_ready_ = false;
    size_t tdt_emit_index_ = 0;
    std::vector<TDTToken> tdt_tokens_;
    size_t last_input_token_count_ = 0;
    std::unique_ptr<npu::NPUEncoder> npu_encoder_;
    bool use_npu_encoder_ = false;
    bool has_cpu_encoder_weights_ = false;
};

} // namespace engine
} // namespace cactus

#include "gemma4/model_gemma4.h"
