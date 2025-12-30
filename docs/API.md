# Whisper Stream Server API Reference

This document describes the WebSocket protocol for communicating with the whisper-stream-server.

## Connection

### Endpoint

```
ws://localhost:9090
```

Or with a custom path (server accepts any path):
```
ws://localhost:9090/transcribe
ws://localhost:9090/api/v1/stream
```

### Authentication

If the server is started with `--token SECRET`, all WebSocket connections must include the token as a query parameter:

```
ws://host:port?token=SECRET
```

Connections without a valid token receive HTTP 401 and are rejected before the WebSocket handshake completes.

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://192.168.1.50:9090?token=my_secret_token');
```

**Example (wscat):**
```bash
wscat -c 'ws://localhost:9090?token=my_secret_token'
```

### Connection Flow

1. Client connects to WebSocket endpoint (with `?token=...` if auth enabled)
2. Server creates session (no whisper context assigned yet)
3. Server sends `ready` message
4. Client streams binary PCM audio
5. When VAD detects speech start → server leases a whisper context
6. Server sends `partial` messages during speech
7. When VAD detects speech end → server sends `final` message, releases context
8. Cycle repeats for next utterance

**Note:** The `ready` message indicates the session is created and ready to receive audio. It does NOT mean a whisper context is allocated. Contexts are leased on-demand when speech is detected, allowing many idle connections without exhausting the context pool.

### Example Connection (JavaScript)

```javascript
// Without authentication
const ws = new WebSocket('ws://localhost:9090');

// With authentication
const ws = new WebSocket('ws://localhost:9090?token=YOUR_SECRET');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log('Received:', msg);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

## HTTP Batch Transcription

### Endpoint

```
POST http://localhost:9090/v1/transcribe
```

### Authentication

If the server is started with `--token SECRET`, all HTTP batch requests must include:

```
Authorization: Bearer SECRET
```

### Request

Headers:
```
Content-Type: audio/wav
Authorization: Bearer <token>   (required if --token is set)
```

Body:
- Raw WAV bytes (PCM16 mono). Any sample rate is accepted; the server resamples to 16kHz.

### Response (200)

```json
{
  "text": "And so my fellow Americans, ask not what your country can do for you.",
  "duration_ms": 11234,
  "processing_ms": 847,
  "words": [
    {"word": "And", "start_ms": 120, "end_ms": 280, "probability": 0.98}
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Full transcript |
| `duration_ms` | number | Audio duration after resampling |
| `processing_ms` | number | Server-side processing time |
| `words` | array | Word timings (10ms resolution) |
| `words[].word` | string | Word text |
| `words[].start_ms` | number | Word start in ms |
| `words[].end_ms` | number | Word end in ms |
| `words[].probability` | number | Geometric mean of token probs |

### Error Responses

| Status | Meaning | Notes |
|--------|---------|-------|
| 401 | Unauthorized | Missing/invalid bearer token |
| 413 | Payload Too Large | Exceeds `--max-upload-size` |
| 415 | Unsupported Media Type | Content-Type must be `audio/wav` |
| 422 | Unprocessable Entity | Invalid WAV or duration too long |
| 503 | Service Unavailable | No context available before timeout |

### Limits and Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max-upload-size` | 25 | Max upload size in MB |
| `--max-audio-duration` | 300000 | Max duration in ms |
| `--batch-timeout` | 10000 | Context wait timeout in ms |

### Logging

Batch requests log a short summary at info level. Set `INFO_LEVEL=debug` for verbose logs
including resampling and inference timing.

### Example (curl)

```bash
curl -X POST http://localhost:9090/v1/transcribe \
  -H "Content-Type: audio/wav" \
  -H "Authorization: Bearer YOUR_SECRET" \
  --data-binary @recording.wav
```

## Message Protocol

### Direction: Server → Client

All server messages are JSON text frames.

#### Ready Message

Sent immediately after connection when a context is successfully acquired.

```json
{
  "type": "ready",
  "model": "models/ggml-base.en.bin",
  "contexts": 2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"ready"` |
| `model` | string | Path to loaded model |
| `contexts` | number | Total contexts in pool |

#### Partial Message

Sent during speech with the current transcription. Updates frequently as more audio is processed.

```json
{
  "type": "partial",
  "text": "Hello how are you"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"partial"` |
| `text` | string | Current transcription (may change) |

**Note**: Partials replace each other. Only display the most recent partial.

#### Final Message

Sent when VAD detects speech has ended (silence threshold exceeded). Contains the complete transcription for the utterance.

```json
{
  "type": "final",
  "text": "Hello, how are you doing today?"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"final"` |
| `text` | string | Finalized transcription |

**VAD Behavior**: When VAD is enabled, the server tracks speech state:
- `IDLE` → `SPEAKING`: Speech detected above threshold
- `SPEAKING` → `ENDING`: Silence exceeds `--vad-silence` duration (default: 1000ms)
- `ENDING` → emits `final` message and returns to `IDLE`

Finals are only emitted when VAD is enabled (`--vad-model` specified).

#### Error Message

Sent when an error occurs.

```json
{
  "type": "error",
  "message": "No available contexts"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"error"` |
| `message` | string | Human-readable error description |

### Direction: Client → Server

Audio data is sent as binary WebSocket frames.

#### Audio Format

| Property | Value |
|----------|-------|
| Encoding | 16-bit signed integer (little-endian) |
| Sample Rate | 16000 Hz |
| Channels | 1 (mono) |
| Byte Order | Little-endian |

#### Chunk Size

Recommended: 100-200ms of audio per frame
- 100ms = 1600 samples = 3200 bytes
- 200ms = 3200 samples = 6400 bytes

Smaller chunks = lower latency, more overhead
Larger chunks = higher latency, less overhead

#### Example: Sending Audio (JavaScript)

```javascript
// Assuming you have Int16Array audio data
const audioChunk = new Int16Array(1600); // 100ms at 16kHz

// Send as binary
ws.send(audioChunk.buffer);
```

## Client Implementation Guide

### 1. Basic Client Structure

```typescript
interface ServerMessage {
  type: 'ready' | 'partial' | 'final' | 'error';
  text?: string;
  message?: string;
  model?: string;
  contexts?: number;
}

class WhisperClient {
  private ws: WebSocket | null = null;

  connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        // Don't resolve yet - wait for 'ready' message
      };

      this.ws.onmessage = (event) => {
        const msg: ServerMessage = JSON.parse(event.data);

        switch (msg.type) {
          case 'ready':
            resolve();
            break;
          case 'partial':
            this.onPartial(msg.text!);
            break;
          case 'final':
            this.onFinal(msg.text!);
            break;
          case 'error':
            this.onError(new Error(msg.message));
            break;
        }
      };

      this.ws.onerror = () => reject(new Error('Connection failed'));
      this.ws.onclose = () => this.onClose();
    });
  }

  sendAudio(pcm: Int16Array): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(pcm.buffer);
    }
  }

  disconnect(): void {
    this.ws?.close();
  }

  // Override these
  onPartial(text: string): void {}
  onFinal(text: string): void {}
  onError(err: Error): void {}
  onClose(): void {}
}
```

### 2. Audio Capture (Browser)

```typescript
async function captureAudio(onChunk: (pcm: Int16Array) => void) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: { ideal: 16000 },
      echoCancellation: true,
      noiseSuppression: true,
    }
  });

  const audioContext = new AudioContext({ sampleRate: 16000 });
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (event) => {
    const float32 = event.inputBuffer.getChannelData(0);
    const int16 = float32ToInt16(float32);
    onChunk(int16);
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  return () => {
    stream.getTracks().forEach(t => t.stop());
    audioContext.close();
  };
}

function float32ToInt16(float32: Float32Array): Int16Array {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16;
}
```

### 3. Resampling (if browser rate != 16kHz)

```typescript
function resample(
  input: Float32Array,
  fromRate: number,
  toRate: number
): Float32Array {
  if (fromRate === toRate) return input;

  const ratio = fromRate / toRate;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const srcIndex = i * ratio;
    const srcFloor = Math.floor(srcIndex);
    const srcCeil = Math.min(srcFloor + 1, input.length - 1);
    const t = srcIndex - srcFloor;

    // Linear interpolation
    output[i] = input[srcFloor] * (1 - t) + input[srcCeil] * t;
  }

  return output;
}
```

## Error Handling

### Connection Errors

| Scenario | Behavior |
|----------|----------|
| Server not running | `onerror` fires, connection fails |
| All contexts busy | `ready` not sent, then error message |
| Invalid audio format | Server ignores, no response |

### Reconnection Strategy

```typescript
async function connectWithRetry(url: string, maxRetries = 5) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await client.connect(url);
      return;
    } catch (err) {
      console.log(`Attempt ${i + 1} failed, retrying...`);
      await sleep(1000 * Math.pow(2, i)); // Exponential backoff
    }
  }
  throw new Error('Failed to connect after retries');
}
```

## Rate Limiting

The server does not implement rate limiting. For production:
- Limit connections per IP
- Limit audio data rate per connection
- Use a reverse proxy (nginx) for rate limiting

## Security Considerations

### Built-in Security

- **Token authentication**: Use `--token SECRET` to require authentication
- **Host binding**: Default `0.0.0.0`, can restrict with `--host 127.0.0.1`

### Current Limitations

- **No encryption**: Plain WebSocket (ws://) - use TLS proxy for production
- **No input validation**: Server trusts audio format

### Production Recommendations

1. **Enable token auth**: Always use `--token` in production
   ```bash
   ./whisper-stream-server --token YOUR_SECRET ...
   ```

2. **Use TLS**: Deploy behind nginx/Caddy with SSL, or use Cloudflare Tunnel
   ```
   wss://your-domain.com/transcribe?token=xxx
   ```

3. **Rate limit**: nginx `limit_conn` and `limit_req`

4. **Validate origin**: Check `Origin` header via reverse proxy

## Protocol Versioning

The current protocol is version 1 (implicit). Current features:
- Real-time partial transcriptions
- VAD-gated final transcriptions (when VAD model loaded)
- Event-driven message delivery (messages flush immediately, not dependent on incoming audio)

Future versions may add:

```json
{
  "type": "ready",
  "version": 2,
  "model": "...",
  "features": ["vad", "timestamps", "speaker-id"]
}
```

## Testing with wscat

```bash
# Install
npm install -g wscat

# Connect (no auth)
wscat -c ws://localhost:9090

# Connect (with auth token)
wscat -c 'ws://localhost:9090?token=YOUR_SECRET'

# You'll receive the ready message
< {"type":"ready","model":"models/ggml-base.en.bin","contexts":2}

# Send audio (not practical via wscat, use browser client)
```

## cURL Testing (HTTP upgrade)

```bash
# Check if server is listening (will fail upgrade but confirms port)
curl -v --include \
  --header "Connection: Upgrade" \
  --header "Upgrade: websocket" \
  --header "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
  --header "Sec-WebSocket-Version: 13" \
  http://localhost:9090/
```
