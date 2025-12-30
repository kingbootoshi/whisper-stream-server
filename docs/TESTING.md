# Testing Guide

This document describes the test suite for whisper-stream-server.

## Overview

The test suite has three layers:

| Layer | Language | Framework | Purpose |
|-------|----------|-----------|---------|
| Unit | C++ | Catch2 | Test individual components in isolation |
| Integration | C++ | Catch2 | Test Whisper transcription with real models |
| E2E | TypeScript | Vitest | Test full WebSocket streaming pipeline |

## Running Tests

```bash
# Run all tests
./scripts/run_tests.sh

# Run only unit tests (no model required)
./scripts/run_tests.sh unit

# Run only integration tests (requires whisper model)
./scripts/run_tests.sh int

# Run only E2E tests (requires running server)
./scripts/run_tests.sh e2e
```

### Prerequisites

- **Unit tests**: No dependencies, always runnable
- **Integration tests**: Requires `../whisper.cpp/models/ggml-base.en.bin`
- **E2E tests**: Requires server running on port 9090

## Test Structure

```
tests/
├── unit/
│   ├── test_audio_buffer.cpp      # AudioBuffer class
│   └── test_vad_state_machine.cpp # VAD state transitions
├── integration/
│   └── test_transcription.cpp     # Whisper inference
├── e2e/
│   ├── package.json
│   ├── vitest.config.ts
│   ├── utils/
│   │   ├── WavLoader.ts           # WAV file loading
│   │   └── TestClient.ts          # WebSocket client
│   └── tests/
│       ├── batch-transcribe.test.ts
│       ├── connection.test.ts
│       ├── streaming.test.ts
│       └── vad-boundaries.test.ts
└── fixtures/
    └── jfk.wav                    # Test audio (symlink)
```

## Unit Tests

### AudioBuffer (`test_audio_buffer.cpp`)

Tests the thread-safe ring buffer that accumulates incoming PCM audio.

| Test | What It Validates |
|------|-------------------|
| `push_increments_size` | Samples accumulate correctly |
| `max_capacity_enforced` | Buffer caps at 30 seconds, drops oldest |
| `int16_to_float_conversion` | 32767→~1.0, -32768→~-1.0, 0→0.0 |
| `getLastMs_returns_correct_slice` | Returns correct tail for VAD |
| `getAll_returns_complete_buffer` | All samples retrievable |
| `clear_empties_buffer` | Buffer empties completely |
| `hasMinDuration_threshold` | Duration checks work |
| `thread_safety_*` | Concurrent push/read doesn't crash |

**Why it matters**: Foundation of audio ingestion. Bugs here corrupt audio before Whisper sees it.

### VAD State Machine (`test_vad_state_machine.cpp`)

Tests the speech detection state machine without requiring Whisper models.

| Test | What It Validates |
|------|-------------------|
| `idle_to_speaking_on_speech` | Speech triggers SPEAKING state |
| `speaking_updates_last_speech_ms` | Timestamps update correctly |
| `speaking_to_ending_after_silence` | 1000ms silence → ENDING |
| `short_utterance_returns_to_idle` | Noise rejection works |
| `ending_to_speaking_on_resume` | Interruption handling |
| `silence_999ms_stays_speaking` | Boundary condition |
| `multiple_utterance_cycle` | State resets properly |

**State Machine:**
```
IDLE ──[speech]──> SPEAKING ──[1000ms silence]──> ENDING ──[emit final]──> IDLE
                       ↑                              │
                       └────[speech resumes]──────────┘
```

**Why it matters**: Controls when `final` transcripts are emitted. Bugs cause fragmented or missing transcripts.

## Integration Tests

### Transcription (`test_transcription.cpp`)

Tests actual Whisper inference using the JFK audio fixture.

| Test | What It Validates |
|------|-------------------|
| `load_wav_file` | WAV parsing handles extra chunks |
| `transcribe_jfk_contains_keywords` | Output has "ask", "country" |
| `transcribe_empty_audio` | Silent audio → empty result |
| `context_initialization` | Model loads/unloads cleanly |

**Ground Truth**: JFK clip (~11 seconds)
- Expected: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

**Why it matters**: Validates transcription quality. If this fails, model or inference is broken.

## E2E Tests

### Connection (`connection.test.ts`)

Tests basic WebSocket connectivity.

| Test | What It Validates |
|------|-------------------|
| `receives ready on connect` | Server sends `{"type":"ready"}` |
| `can disconnect cleanly` | No errors on close |
| `can reconnect after disconnect` | Session cleanup works |
| `multiple simultaneous clients` | Context pool handles concurrency |

**Why it matters**: If these fail, no client can connect.

### Streaming (`streaming.test.ts`)

Tests audio streaming and transcription.

| Test | What It Validates |
|------|-------------------|
| `receives partials when streaming` | Partial transcripts arrive |
| `partial contains expected keywords` | Content is correct |
| `works with 50ms/100ms/200ms chunks` | Chunk size flexibility |
| `handles rapid streaming` | Server survives burst traffic |
| `streams at real-time speed` | 1x playback works |

**Why it matters**: Validates the core streaming use case.

### VAD Boundaries (`vad-boundaries.test.ts`)

Tests voice activity detection and final transcript emission.

| Test | What It Validates |
|------|-------------------|
| `final after silence` | Finals emitted after speech ends |
| `final contains full transcript` | Complete utterance captured |
| `partials during speech` | No premature finals |
| `no finals during silence` | No spurious output |
| `multiple utterances` | Each utterance gets a final |

**Why it matters**: Tests the full VAD-gated pipeline. Catches bugs like empty finals.

### Batch Transcribe (`batch-transcribe.test.ts`)

Tests the HTTP batch transcription endpoint.

| Test | What It Validates |
|------|-------------------|
| `returns text + word timestamps for JFK WAV` | JSON shape + word timing sanity |

**Why it matters**: Validates WAV upload path, context pool use, and word timestamp output.

## Test Fixtures

### `jfk.wav`

- **Duration**: ~11 seconds
- **Format**: 16-bit PCM, mono, 16kHz
- **Content**: JFK's "Ask not what your country can do for you" speech
- **Location**: `samples/jfk.wav` (symlinked to `tests/fixtures/`)

This is the ground truth for transcription tests. The expected output is well-known, making it easy to verify correctness.

## Adding New Tests

### C++ Unit Test

```cpp
// tests/unit/test_new_component.cpp
#include <catch2/catch_test_macros.hpp>
#include "new_component.hpp"

TEST_CASE("Component does something", "[component]") {
    // Arrange
    NewComponent c;

    // Act
    auto result = c.doSomething();

    // Assert
    REQUIRE(result == expected);
}
```

Add to `CMakeLists.txt`:
```cmake
add_executable(test_new_component tests/unit/test_new_component.cpp)
target_link_libraries(test_new_component PRIVATE Catch2::Catch2WithMain)
add_test(NAME NewComponent COMMAND test_new_component)
```

### TypeScript E2E Test

```typescript
// tests/e2e/tests/new-feature.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { TestClient } from '../utils/TestClient.js';

describe('New Feature', () => {
  let client: TestClient;

  beforeEach(async () => {
    client = new TestClient();
    await client.connect();
    await client.waitForReady();
  });

  afterEach(() => {
    client.disconnect();
  });

  it('does something', async () => {
    // Test implementation
  });
});
```

## Regression Coverage

These tests catch:

| Bug Type | Caught By |
|----------|-----------|
| Audio buffer corruption | `test_audio_buffer` |
| Thread safety issues | `thread_safety_*` tests |
| Wrong speech boundaries | `test_vad_state_machine` |
| Empty final transcripts | `vad-boundaries.test.ts` |
| Model loading failures | `test_transcription` |
| WebSocket protocol issues | `connection.test.ts` |
| Chunk size incompatibility | `streaming.test.ts` |

## CI Integration

To run tests in CI:

```yaml
# Example GitHub Actions
- name: Run unit tests
  run: ./scripts/run_tests.sh unit

- name: Run integration tests
  run: ./scripts/run_tests.sh int
  env:
    MODEL_PATH: ./models/ggml-base.en.bin

- name: Start server and run E2E
  run: |
    ./scripts/run.sh &
    sleep 5
    ./scripts/run_tests.sh e2e
```
