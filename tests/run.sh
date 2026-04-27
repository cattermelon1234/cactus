#!/bin/bash

echo "Running Cactus test suite..."
echo "============================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_MODEL="google/gemma-4-E2B-it"
MODEL_NAME="$DEFAULT_MODEL"
ANDROID_MODE=false
IOS_MODE=false
NO_REBUILD=false
ONLY_EXEC=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --android)
            ANDROID_MODE=true
            shift
            ;;
        --ios)
            IOS_MODE=true
            shift
            ;;
        --no-rebuild)
            NO_REBUILD=true
            shift
            ;;
        --only)
            ONLY_EXEC="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        # Accept and ignore removed model flags for backwards compatibility
        --transcribe_model|--whisper_model|--vad_model|--diarize_model|--embed_speaker_model)
            shift 2
            ;;
        --exhaustive)
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model <name>            Model to use for tests (default: $DEFAULT_MODEL)"
            echo "  --precision <type>        Precision for model conversion (MIXED, FP16, INT8, INT4)"
            echo "  --android                 Run tests on Android device or emulator"
            echo "  --ios                     Run tests on iOS device or simulator"
            echo "  --no-rebuild              Skip building cactus library and tests"
            echo "  --only <test_name>        Only run the specified test (llm, vlm, stt, embed, rag, graph, index, kernel, kv_cache, performance)"
            echo "  --help, -h                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "Using model: $MODEL_NAME"
if [ -n "$PRECISION" ]; then
    echo "Using precision: $PRECISION"
    PRECISION_FLAG="--precision $PRECISION"
else
    PRECISION_FLAG=""
fi

echo ""
echo "Step 1: Downloading model weights..."
if ! cactus download "$MODEL_NAME" $PRECISION_FLAG; then
    echo "Failed to download model weights"
    exit 1
fi

if [ "$ANDROID_MODE" = true ]; then
    export CACTUS_TEST_ONLY="$ONLY_EXEC"
    exec "$SCRIPT_DIR/android/run.sh" "$MODEL_NAME"
fi

if [ "$IOS_MODE" = true ]; then
    export CACTUS_TEST_ONLY="$ONLY_EXEC"
    exec "$SCRIPT_DIR/ios/run.sh" "$MODEL_NAME"
fi

if [ "$NO_REBUILD" = false ]; then
    echo ""
    echo "Step 2: Building Cactus library..."
    if ! cactus build; then
        echo "Failed to build cactus library"
        exit 1
    fi

    echo ""
    echo "Step 3: Building tests..."
    cd "$PROJECT_ROOT/tests"

    rm -rf build
    mkdir -p build
    cd build

    if ! cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF > /dev/null 2>&1; then
        echo "Failed to configure tests"
        exit 1
    fi

    if ! make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4); then
        echo "Failed to build tests"
        exit 1
    fi
else
    echo "Skipping build (--no-rebuild)"
    cd "$PROJECT_ROOT/tests/build"
fi

echo ""
echo "Step 4: Running tests..."
echo "------------------------"

MODEL_DIR=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

export CACTUS_TEST_MODEL="$PROJECT_ROOT/weights/$MODEL_DIR"
export CACTUS_TEST_ASSETS="$PROJECT_ROOT/tests/assets"
export CACTUS_INDEX_PATH="$PROJECT_ROOT/tests/assets"

echo "Using model path: $CACTUS_TEST_MODEL"
echo "Using assets path: $CACTUS_TEST_ASSETS"

echo "Discovering test executables..."
test_executables=($(find . -maxdepth 1 -name "test_*" -type f | sort))

executable_tests=()
for test_file in "${test_executables[@]}"; do
    if [ -x "$test_file" ]; then
        executable_tests+=("$test_file")
    fi
done

if [ ${#executable_tests[@]} -eq 0 ]; then
    echo "No test executables found!"
    exit 1
fi

test_executables=("${executable_tests[@]}")

if [ -n "$ONLY_EXEC" ]; then
    allowed=()
    for test_file in "${executable_tests[@]}"; do
        test_name=$(basename "$test_file" | sed 's/^test_//')
        allowed+=("$test_name")
    done

    ok=false
    for a in "${allowed[@]}"; do
        if [ "$a" = "$ONLY_EXEC" ]; then
            ok=true
            break
        fi
    done
    if [ "$ok" = false ]; then
        echo "Unknown test name: $ONLY_EXEC"
        echo "Allowed: ${allowed[*]}"
        exit 1
    fi

    target="./test_$ONLY_EXEC"
    if [ ! -f "$target" ] || [ ! -x "$target" ]; then
        echo "Could not find or execute test: $target"
        exit 1
    fi

    test_executables=("$target")
fi

echo "Found ${#test_executables[@]} test executable(s)"

for executable in "${test_executables[@]}"; do
    exec_name=$(basename "$executable")
    ./"$exec_name"
done
