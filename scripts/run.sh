#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SERVER_DIR/build"
BINARY="$BUILD_DIR/whisper-stream-server"
MODELS_DIR="$SERVER_DIR/models"

INFO_LEVEL="${INFO_LEVEL:-debug}"
export INFO_LEVEL

# Default model paths
DEFAULT_MODEL="$MODELS_DIR/ggml-base.en.bin"
DEFAULT_VAD_MODEL="$MODELS_DIR/ggml-silero-vad.bin"

# Also check whisper.cpp sibling directory as fallback
WHISPER_CPP_DIR="$SERVER_DIR/../whisper.cpp"

if [ ! -f "$BINARY" ]; then
    echo "Error: Server binary not found at $BINARY"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

# Check for whisper model
MODEL="${MODEL:-$DEFAULT_MODEL}"
if [ ! -f "$MODEL" ]; then
    # Try whisper.cpp sibling directory
    if [ -f "$WHISPER_CPP_DIR/models/ggml-base.en.bin" ]; then
        MODEL="$WHISPER_CPP_DIR/models/ggml-base.en.bin"
    else
        echo "Error: Whisper model not found at $MODEL"
        echo ""
        echo "Download with:"
        echo "  mkdir -p models"
        echo "  curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -o models/ggml-base.en.bin"
        exit 1
    fi
fi

# Check for VAD model (required)
VAD_MODEL="${VAD_MODEL:-$DEFAULT_VAD_MODEL}"
if [ ! -f "$VAD_MODEL" ]; then
    # Try whisper.cpp sibling directory
    for vad_file in "$WHISPER_CPP_DIR/models/ggml-silero-v"*.bin; do
        if [ -f "$vad_file" ]; then
            VAD_MODEL="$vad_file"
            break
        fi
    done
fi

if [ ! -f "$VAD_MODEL" ]; then
    echo "Error: VAD model not found at $VAD_MODEL"
    echo ""
    echo "Download with:"
    echo "  mkdir -p models"
    echo "  curl -L https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin -o models/ggml-silero-vad.bin"
    exit 1
fi

echo ""
echo "Starting whisper-stream-server..."
echo "Model: $MODEL"
echo "VAD:   $VAD_MODEL"
echo "Host:  ${HOST:-0.0.0.0}"
echo "Port:  ${PORT:-9090}"
if [ -n "$TOKEN" ]; then
    echo "Auth:  Token enabled"
fi
echo ""

# Build command
CMD="$BINARY --model $MODEL --vad-model $VAD_MODEL --port ${PORT:-9090} --contexts ${CONTEXTS:-2} --threads ${THREADS:-4}"

# Add optional args
if [ -n "$HOST" ]; then
    CMD="$CMD --host $HOST"
fi

if [ -n "$TOKEN" ]; then
    CMD="$CMD --token $TOKEN"
fi

if [ -n "$VAD_THRESHOLD" ]; then
    CMD="$CMD --vad-threshold $VAD_THRESHOLD"
fi

if [ -n "$VAD_SILENCE" ]; then
    CMD="$CMD --vad-silence $VAD_SILENCE"
fi

# Execute with any additional args
exec $CMD "$@"
