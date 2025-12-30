# Local Mac Audio Transcription

Real-time speech-to-text transcription server using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with WebSocket streaming. Optimized for Apple Silicon with Metal GPU acceleration.

## Features

- **Real-time streaming transcription** via WebSocket
- **Batch transcription** via HTTP WAV upload (`POST /v1/transcribe`)
- **Voice Activity Detection (VAD)** - automatically detects speech end and emits final transcripts
- **Metal GPU acceleration** on Apple Silicon (M1/M2/M3/M4)
- **Multi-client support** with context pooling (configurable parallel sessions)
- **Low latency** (~300-500ms inference on M-series chips)
- **24/7 operation** - designed for always-on deployments via launchd

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- CMake 3.14+
- Xcode Command Line Tools

```bash
# Install build tools
brew install cmake
xcode-select --install

# Clone this repo
git clone https://github.com/kingbootoshi/local-mac-audio-transcription.git
cd local-mac-audio-transcription
```

### Build

```bash
./scripts/build.sh
```

The build script uses CMake FetchContent to automatically download whisper.cpp. No manual cloning required.

### Run

```bash
./scripts/run.sh
```

Or with custom options:
```bash
./build/whisper-stream-server \
  --model models/ggml-base.en.bin \
  --vad-model models/ggml-silero-vad.bin \
  --port 9090 \
  --contexts 2
```

### Test with Browser Client

```bash
cd examples/web-client
npm install
npm run dev
```

Open http://localhost:5173, click **Connect**, then **Start Recording**.

## Project Structure

```
local-mac-audio-transcription/
├── src/                        # C++ server source
│   ├── main.cpp               # Entry point, WebSocket handlers
│   ├── whisper_server.cpp     # Core transcription + VAD logic
│   ├── whisper_server.hpp
│   ├── audio_buffer.cpp       # Thread-safe audio buffer
│   ├── audio_buffer.hpp
│   └── json.hpp               # nlohmann/json (auto-downloaded)
│
├── scripts/
│   ├── build.sh               # Build script
│   └── run.sh                 # Run script
│
├── examples/                   # Example clients
│   └── web-client/            # TypeScript browser client
│
├── install/                    # Mac deployment
│   ├── install.sh             # Install as launchd service
│   ├── uninstall.sh
│   └── com.whisper.stream-server.plist
│
├── docs/
│   ├── ARCHITECTURE.md        # Server design deep-dive
│   ├── API.md                 # WebSocket protocol reference
│   └── CPP_GUIDE.md           # C++ learning guide
│
├── CMakeLists.txt
└── README.md
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Path to whisper model |
| `--vad-model` | (required) | Path to VAD model |
| `--port` | `9090` | WebSocket server port |
| `--host` | `0.0.0.0` | Bind address |
| `--token` | (none) | Authentication token for WebSocket connections |
| `--contexts` | `2` | Number of parallel transcription contexts |
| `--threads` | `4` | CPU threads per inference |
| `--step` | `500` | Inference interval (ms) |
| `--length` | `5000` | Audio context window (ms) |
| `--keep` | `200` | Overlap between windows (ms) |
| `--vad-threshold` | `0.5` | Voice activity detection threshold |
| `--vad-silence` | `1000` | Silence duration to trigger final (ms) |
| `--language` | `en` | Language code |
| `--no-gpu` | - | Disable Metal GPU |
| `--max-upload-size` | `25` | Max batch upload size in MB |
| `--max-audio-duration` | `300000` | Max batch duration in ms |
| `--batch-timeout` | `10000` | Batch context wait timeout (ms) |

### Logging

Set `INFO_LEVEL=debug` to enable verbose logs (batch inference timing, resample info).
Default is `info`, but `./scripts/run.sh` sets `INFO_LEVEL=debug` unless overridden.

## WebSocket Protocol

### Client → Server
- **Binary frames**: 16-bit signed PCM audio at 16kHz mono

### Server → Client
```json
{ "type": "ready", "model": "base.en", "contexts": 2 }
{ "type": "partial", "text": "Hello how are" }
{ "type": "final", "text": "Hello, how are you?" }
{ "type": "error", "message": "..." }
```

## Security

### Token Authentication

For any deployment accessible over a network, use token authentication:

**Server:**
```bash
./build/whisper-stream-server --token YOUR_SECRET_TOKEN ...
```

**Client:**
```javascript
const ws = new WebSocket('ws://host:port?token=YOUR_SECRET_TOKEN');
```

Connections without a valid token are rejected with HTTP 401.

## Mac Mini Deployment (24/7)

### Install as System Service

```bash
# Build first
./scripts/build.sh

# Install (downloads models automatically)
sudo ./install/install.sh --token YOUR_SECRET

# With all options
sudo ./install/install.sh --token YOUR_SECRET --port 9090 --contexts 4
```

The install script will:
1. Build the binary if not already built
2. Download the whisper model (~148 MB) from Hugging Face
3. Download the VAD model (~1 MB) from Hugging Face
4. Install binary to `/usr/local/bin/`
5. Install models to `/usr/local/share/whisper/`
6. Create and start a launchd service

### Service Management

```bash
# Check status
sudo launchctl list | grep whisper

# View logs
tail -f /usr/local/var/log/whisper-stream-server.log

# Stop
sudo launchctl unload /Library/LaunchDaemons/com.whisper.stream-server.plist

# Start
sudo launchctl load /Library/LaunchDaemons/com.whisper.stream-server.plist

# Verify port is listening
lsof -nP -iTCP:9090 -sTCP:LISTEN
```

### Test Connection

```bash
# If token auth enabled
wscat -c 'ws://localhost:9090?token=YOUR_SECRET'

# If no token
wscat -c ws://localhost:9090
```

## Memory Usage

| Contexts | Model | RAM |
|----------|-------|-----|
| 2 | base.en | ~850 MB |
| 4 | base.en | ~1.6 GB |
| 2 | small.en | ~1.8 GB |
| 4 | small.en | ~3.5 GB |

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Threading, context pooling, VAD
- [API.md](docs/API.md) - WebSocket protocol, client examples
- [CPP_GUIDE.md](docs/CPP_GUIDE.md) - C++ learning guide

## License

MIT
