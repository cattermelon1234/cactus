<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

```
┌─────────────────┐     Energy-efficient inference engine for running AI on mobile devices 
│  Cactus Engine  │ ←── OpenAI compatible APIs for C/C++, Swift, Kotlin, Flutter & React-Native
└─────────────────┘     Supports tool call, auto RAG, NPU, INT4, and cloud handoff for complex tasks
         │
┌─────────────────┐     Zero-copy computation graph, think PyTorch for mobile devices
│  Cactus Graph   │ ←── You can implement custom models directly using this
└─────────────────┘     Highly optimised for RAM & lossless weight quantisation 
         │
┌─────────────────┐     Low-level ARM-specific SIMD kernels (Apple, Snapdragon, Google, Exynos, MediaTek & Raspberry Pi)
│ Cactus Kernels  │ ←── Optimised Matrix Multiplication & n
└─────────────────┘     Custom attention kernels with KV-Cache Quantisation, chunked prefill, streaming LLM, etc.
```


## Cactus Engine

```cpp
#include cactus.h

cactus_model_t model = cactus_init(
    "path/to/weight/folder",
    "path to txt or dir of txts for auto-rag",
);

const char* messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Henry Ndubuaku"}
])";

const char* options = R"({
    "max_tokens": 50,
    "stop_sequences": ["<|im_end|>"]
})";

char response[4096];
int result = cactus_complete(
    model,                            // model handle from cactus_init
    messages,                         // JSON array of chat messages
    response,                         // buffer to store response JSON
    sizeof(response),                 // size of response buffer
    options,                          // optional: generation options (nullptr for defaults)
    nullptr,                          // optional: tools JSON for function calling 
    nullptr,                          // optional: streaming callback fn(token, id, user_data)
    nullptr                           // optional: user data passed to callback
);
```
Example response from Gemma3-270m
```json
{
    "success": true,                 // when successfully generated
    "error": null,                   // returns specific errors if success = false
    "cloud_handoff": false,          // true when response is generated with cloud model
    "response": "Hi there!",         // null when error is not null
    "function_calls": [],            // parsed to [{"name":"set_alarm","arguments":{"hour":"10","minute":"0"}}]
    "confidence": 0.8193,            // how confident the model is with its locally generated response
    "time_to_first_token_ms": 45.23, // latency (time to first token)
    "total_time_ms": 163.67,         // total execution time
    "prefill_tps": 1621.89,          // prefill tokens per second
    "decode_tps": 168.42,            // decode tokens per second
    "ram_usage_mb": 245.67,          // current process RAM usage in MB
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

## Cactus Graph

```cpp
#include cactus.h

CactusGraph graph;
auto a = graph.input({2, 3}, Precision::FP16);
auto b = graph.input({3, 4}, Precision::INT8);

auto x1 = graph.matmul(a, b, false);
auto x2 = graph.transpose(x1);
auto result = graph.matmul(b, x2, true);

float a_data[6] = {1.1f, 2.3f, 3.4f, 4.2f, 5.7f, 6.8f};
float b_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

graph.set_input(a, a_data, Precision::FP16);
graph.set_input(b, b_data, Precision::INT8);

graph.execute();
void* output_data = graph.get_output(result);

graph.hard_reset(); 
```

## Benchmark (missing latency = no NPU support yet)

**High-End Devices**
| Device | LFM2.5-1.2B-INT4<br>(1k-Prefill/100-Decode) | LFM2.5-VL-1.6B-INT4<br>(256px-Latency & Decode) | Parakeet-1.1B-INT4<br>(30s-audio-Latency & Decode)
|--------|--------|--------|----------|
| Mac M4 Pro (Highend) | 582tps/100tps (76MB RAM) | 0.2s/98tps (87MB RAM) | 0.1s/900k+tps (1GB RAM) |
| iPad/Mac M3 (Budget) | 350tps/60tps (70MB RAM) | 0.3s/69tps (80MB RAM) | 0.3s/800k+tps (102MB RAM) |
| iPhone 17 Pro (Highend) | 327tps/48tps (108MB RAM)| 0.3s/48tps (156MB RAM) | 0.3s/300k+tps (177MB RAM)|
| iPhone 13 Mini (Budget) | 148tps/34tps (1GB RAM) | 0.3s/35tps (1.2GBMB RAM) | 0.7s/90k+tps (1GB RAM) |
| Galaxy S25 Ultra (Qualcomm 8 Elite) | 255tps/37tps (1.5GB RAM) | -/34tps (2GB RAM) | -/250k+tps (1.8GBG RAM) |
| Pixel 6a (Budget Google Tensor) | 70tps/15tps (1GB RAM)| -/15tps (1.5GB RAM) | - /17k+tps (1GB RAM)|
| Galaxy A17 5G (Budget Exxynox) | 32tps/10tps (727MB RAM) | -/11tps (727MB RAM) | -/40k+tps (809MB RAM) |
| CMF Phone 2 Pro (Budget Mediatek) | - | - | - |
| Raspberry Pi 5 (IoT) | - | - | - |

## Supported Models                                                                                                                                                     
                                                                                                                                                                          
| Model | Features |                                                                                                                                             
|-------|----------|                                                                                                                                             
| google/gemma-3-270m-it | completion |                                                                                                                         
| google/functiongemma-270m-it | completion, tools |                                                                                                            
| LiquidAI/LFM2-350M | completion, tools, embed |                                                                                                               
| Qwen/Qwen3-0.6B | completion, tools, embed |                                                                                                                  
| LiquidAI/LFM2-700M | completion, tools, embed |                                                                                                               
| LiquidAI/LFM2-8B-A1B | completion, tools, embed |                                                                                                                
| google/gemma-3-1b-it | completion |                                                                                                                           
| LiquidAI/LFM2.5-1.2B-Thinking | completion, tools, embed |                                                                                                    
| LiquidAI/LFM2.5-1.2B-Instruct | completion, tools, embed |                                                                                                      
| Qwen/Qwen3-1.7B | completion, tools, embed | 
| LiquidAI/LFM2-2.6B | completion, tools, embed |                                                                                                                
| LiquidAI/LFM2-VL-450M | vision, txt & img embed, Apple NPU |                                                                                                            
| LiquidAI/LFM2.5-VL-1.6B | vision, txt & img embed, Apple NPU |                                                                                                               
| UsefulSensors/moonshine-base | transcription, speech embed |                                                                                                         
| openai/whisper-small | transcription, speech embed, Apple NPU |                                                                                                                 
| openai/whisper-medium | transcribe, speech embed, Apple NPU |
| nvidia/parakeet-ctc-0.6b | transcribe, speech embed, Apple NPU |
| nvidia/parakeet-ctc-1.1b | transcribe, speech embed, Apple NPU |
| snakers4/silero-vad | vad |
| nomic-ai/nomic-embed-text-v2-moe | embed |                                                                                                                    
| Qwen/Qwen3-Embedding-0.6B | embed | 

## Using this repo on Mac
```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
```

## Using this repo on Linux (Ubuntu/Debian)

```bash
sudo apt-get install python3 python3-venv python3-pip cmake build-essential libcurl4-openssl-dev
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
```

| Command | Description |
|---------|-------------|
| `cactus auth` | Setup Cactus cloud fallback (optional) (`--status`, `--clear`) |
| `cactus run [model]` | Opens playground (auto downloads model) |
| `cactus download [model]` | Downloads model to `./weights` |
| `cactus convert [model] [dir]` | Converts model, supports LoRA merging (`--lora <path>`) |
| `cactus build` | Builds for ARM (`--apple` or `--android`) |
| `cactus test` | Runs tests (`--ios` / `--android`, `--model [name/path]`, `--transcribe_model [name/path]`, `--only [test_name]`, `--precision`) |
| `cactus transcribe [model]` | Transcribe audio file (`--file`) or live microphone |
| `cactus clean` | Removes build artifacts |
| `cactus --help` | Shows all commands and flags (always run this) |

- Reproduce reported benchmarks with `cactus test --benchmark`
- Plug in any mobule device and add the `--ios` or `--android` flag.
- Mobile devices must be in developer mode.


## Using in your apps 

- [Python for Mac](/python/)
- [Rust SDK](/rust/)
- [React Native SDK](https://github.com/cactus-compute/cactus-react-native)
- [Swift Multiplatform SDK](https://github.com/mhayes853/swift-cactus)
- [Kotlin Multiplatform SDK](https://github.com/cactus-compute/cactus-kotlin)
- [Flutter SDK](https://github.com/cactus-compute/cactus-flutter)

## Try demo apps 

- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)

## Maintaining Organisations
Developed by [Cactus Compute, Inc. (YC S25)](https://cactuscompute.com/), with maintenance from:

1. [UCLA's BruinAI](https://bruinai.org/) 
2. [Char (YC S25)](https://char.com/)
3. [Yale's AI Society](https://www.yale-ai.org/team)
4. [National Unoversity of Singapore's AI Society](https://www.nusaisociety.org/)
5. [UC Irvine's AI@UCI](https://aiclub.ics.uci.edu/)
6. [Imperial College's AI Society](https://www.imperialcollegeunion.org/csp/1391)
7. [University of Pennsylvania's AI@Penn](https://ai-at-penn-main-105.vercel.app/)
8. [University of Michigan Ann-Arbor MSAIL](https://msail.github.io/)
9. [University of Colorado Boulder's AI Club](https://www.cuaiclub.org/)

## Contributing to Cactus

- **C++ Standard**: Use C++20 features where appropriate.
- **Formatting**: Follow the existing code style in the project, one header per folder.
- **Comments**: Avoid comments, make your code read like plain english.
- **AI-Generated Code**: Do not bindly PR AI slop, this codebase is very complex, they miss details.
- **Update docs**: Please update docs when necessary, be intuitive and straightforward. 
- **Keep It Simple**: Do not go beyond the scope of the GH issue, avoid bloated PRs, keep codes lean.
- **Benchmark Your Changes**: Test performance impact, Cactus is performance-critical.
- **Test everything**: A PR that fails to build is the biggest red flag, means it was not tested. 

## Citation

If you use Cactus in your research, please cite it as follows:

```bibtex
@software{cactus,
  title        = {Cactus: AI Inference Engine for Phones & Wearables},
  author       = {Ndubuaku, Henry and Cactus Team},
  url          = {https://github.com/cactus-compute/cactus},
  year         = {2025}
}
```

## Join The Community
- [Reddit Channel](https://www.reddit.com/r/cactuscompute/)
