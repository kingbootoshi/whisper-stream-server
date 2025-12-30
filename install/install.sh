#!/bin/bash
#
# Whisper Stream Server - Installation Script
# Installs the server as a launchd service on macOS
#
# Usage: sudo ./install.sh [options]
#
# Options:
#   --model PATH      Path to whisper model (default: downloads base.en)
#   --vad-model PATH  Path to VAD model (default: downloads silero)
#   --port PORT       Server port (default: 9090)
#   --host ADDRESS    Bind address (default: 0.0.0.0)
#   --token SECRET    Authentication token for connections
#   --contexts N      Number of parallel contexts (default: 2)
#   --info-level LVL  Log level: info|debug (default: info)
#   --max-upload-size MB     Max upload size in MB (default: 25)
#   --max-audio-duration MS  Max batch duration in ms (default: 300000)
#   --batch-timeout MS       Context wait timeout in ms (default: 10000)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH=""
VAD_MODEL_PATH=""
PORT="9090"
HOST="0.0.0.0"
TOKEN=""
CONTEXTS="2"
INFO_LEVEL="info"
MAX_UPLOAD_SIZE=""
MAX_AUDIO_DURATION=""
BATCH_TIMEOUT=""

# Installation paths
INSTALL_BIN="/usr/local/bin"
INSTALL_SHARE="/usr/local/share/whisper"
INSTALL_LOG="/usr/local/var/log"
PLIST_PATH="/Library/LaunchDaemons/com.whisper.stream-server.plist"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --vad-model)
            VAD_MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --contexts)
            CONTEXTS="$2"
            shift 2
            ;;
        --info-level)
            INFO_LEVEL="$2"
            shift 2
            ;;
        --max-upload-size)
            MAX_UPLOAD_SIZE="$2"
            shift 2
            ;;
        --max-audio-duration)
            MAX_AUDIO_DURATION="$2"
            shift 2
            ;;
        --batch-timeout)
            BATCH_TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check for root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

echo -e "${GREEN}=== Whisper Stream Server Installer ===${NC}"
echo ""

# Find the project directory (assuming script is in install/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if binary exists
BINARY_PATH="$PROJECT_DIR/build/whisper-stream-server"
if [[ ! -f "$BINARY_PATH" ]]; then
    echo -e "${YELLOW}Binary not found. Building...${NC}"
    cd "$PROJECT_DIR"

    if [[ ! -f "scripts/build.sh" ]]; then
        echo -e "${RED}Error: build.sh not found. Please build manually first.${NC}"
        exit 1
    fi

    ./scripts/build.sh

    if [[ ! -f "$BINARY_PATH" ]]; then
        echo -e "${RED}Error: Build failed.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}[1/6] Creating directories...${NC}"
mkdir -p "$INSTALL_BIN"
mkdir -p "$INSTALL_SHARE"
mkdir -p "$INSTALL_LOG"

echo -e "${GREEN}[2/6] Installing binary...${NC}"
cp "$BINARY_PATH" "$INSTALL_BIN/whisper-stream-server"
chmod 755 "$INSTALL_BIN/whisper-stream-server"
echo "  Installed: $INSTALL_BIN/whisper-stream-server"

echo -e "${GREEN}[3/6] Installing whisper model...${NC}"
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
        exit 1
    fi
    cp "$MODEL_PATH" "$INSTALL_SHARE/ggml-base.en.bin"
    echo "  Copied from: $MODEL_PATH"
elif [[ -f "$INSTALL_SHARE/ggml-base.en.bin" ]]; then
    echo "  Using existing model at $INSTALL_SHARE/ggml-base.en.bin"
else
    echo -e "${YELLOW}  Downloading base.en model from Hugging Face...${NC}"
    curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" \
        -o "$INSTALL_SHARE/ggml-base.en.bin" \
        --progress-bar
    if [[ ! -f "$INSTALL_SHARE/ggml-base.en.bin" ]]; then
        echo -e "${RED}Error: Failed to download whisper model${NC}"
        exit 1
    fi
fi
echo "  Model: $INSTALL_SHARE/ggml-base.en.bin"

echo -e "${GREEN}[4/6] Installing VAD model...${NC}"
if [[ -n "$VAD_MODEL_PATH" ]]; then
    if [[ ! -f "$VAD_MODEL_PATH" ]]; then
        echo -e "${RED}Error: VAD model not found at $VAD_MODEL_PATH${NC}"
        exit 1
    fi
    cp "$VAD_MODEL_PATH" "$INSTALL_SHARE/ggml-silero-vad.bin"
    echo "  Copied from: $VAD_MODEL_PATH"
elif [[ -f "$INSTALL_SHARE/ggml-silero-vad.bin" ]]; then
    echo "  Using existing VAD model at $INSTALL_SHARE/ggml-silero-vad.bin"
else
    echo -e "${YELLOW}  Downloading Silero VAD model from Hugging Face...${NC}"
    curl -L "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin" \
        -o "$INSTALL_SHARE/ggml-silero-vad.bin" \
        --progress-bar
    if [[ ! -f "$INSTALL_SHARE/ggml-silero-vad.bin" ]]; then
        echo -e "${RED}Error: Failed to download VAD model${NC}"
        exit 1
    fi
fi
echo "  VAD Model: $INSTALL_SHARE/ggml-silero-vad.bin"

echo -e "${GREEN}[5/6] Installing launchd plist...${NC}"

# Build ProgramArguments
PROGRAM_ARGS="        <string>/usr/local/bin/whisper-stream-server</string>
        <string>--model</string>
        <string>/usr/local/share/whisper/ggml-base.en.bin</string>
        <string>--vad-model</string>
        <string>/usr/local/share/whisper/ggml-silero-vad.bin</string>
        <string>--port</string>
        <string>${PORT}</string>
        <string>--host</string>
        <string>${HOST}</string>
        <string>--contexts</string>
        <string>${CONTEXTS}</string>
        <string>--threads</string>
        <string>4</string>"

# Add token if provided
if [[ -n "$TOKEN" ]]; then
    PROGRAM_ARGS="$PROGRAM_ARGS
        <string>--token</string>
        <string>${TOKEN}</string>"
fi

if [[ -n "$MAX_UPLOAD_SIZE" ]]; then
    PROGRAM_ARGS="$PROGRAM_ARGS
        <string>--max-upload-size</string>
        <string>${MAX_UPLOAD_SIZE}</string>"
fi

if [[ -n "$MAX_AUDIO_DURATION" ]]; then
    PROGRAM_ARGS="$PROGRAM_ARGS
        <string>--max-audio-duration</string>
        <string>${MAX_AUDIO_DURATION}</string>"
fi

if [[ -n "$BATCH_TIMEOUT" ]]; then
    PROGRAM_ARGS="$PROGRAM_ARGS
        <string>--batch-timeout</string>
        <string>${BATCH_TIMEOUT}</string>"
fi

# Generate plist with custom settings
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisper.stream-server</string>

    <key>ProgramArguments</key>
    <array>
$PROGRAM_ARGS
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>10</integer>

    <key>WorkingDirectory</key>
    <string>/usr/local/share/whisper</string>

    <key>StandardOutPath</key>
    <string>/usr/local/var/log/whisper-stream-server.log</string>

    <key>StandardErrorPath</key>
    <string>/usr/local/var/log/whisper-stream-server.error.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
        <key>INFO_LEVEL</key>
        <string>${INFO_LEVEL}</string>
    </dict>

    <key>Nice</key>
    <integer>-5</integer>
</dict>
</plist>
EOF

chmod 644 "$PLIST_PATH"
echo "  Installed: $PLIST_PATH"

echo -e "${GREEN}[6/6] Starting service...${NC}"

# Unload if already loaded
launchctl unload "$PLIST_PATH" 2>/dev/null || true

# Load the service
launchctl load "$PLIST_PATH"

# Wait a moment for startup
sleep 2

# Check if running
if launchctl list | grep -q "com.whisper.stream-server"; then
    echo -e "${GREEN}  Service started successfully!${NC}"
else
    echo -e "${YELLOW}  Service may not have started. Check logs:${NC}"
    echo "    tail -f $INSTALL_LOG/whisper-stream-server.log"
    echo "    tail -f $INSTALL_LOG/whisper-stream-server.error.log"
fi

echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Service Details:"
echo "  Binary:   $INSTALL_BIN/whisper-stream-server"
echo "  Model:    $INSTALL_SHARE/ggml-base.en.bin"
echo "  VAD:      $INSTALL_SHARE/ggml-silero-vad.bin"
echo "  Host:     $HOST"
echo "  Port:     $PORT"
echo "  Contexts: $CONTEXTS"
if [[ -n "$TOKEN" ]]; then
    echo "  Auth:     Token required (ws://host:port?token=...)"
else
    echo "  Auth:     None (add --token SECRET for production)"
fi
echo "  Logs:     $INSTALL_LOG/whisper-stream-server.log"
echo ""
echo "Commands:"
echo "  Status:   sudo launchctl list | grep whisper"
echo "  Stop:     sudo launchctl unload $PLIST_PATH"
echo "  Start:    sudo launchctl load $PLIST_PATH"
echo "  Logs:     tail -f $INSTALL_LOG/whisper-stream-server.log"
echo ""
echo "Test with:"
if [[ -n "$TOKEN" ]]; then
    echo "  wscat -c 'ws://localhost:$PORT?token=YOUR_TOKEN'"
else
    echo "  wscat -c ws://localhost:$PORT"
fi
echo ""
