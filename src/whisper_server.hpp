#ifndef WHISPER_SERVER_HPP
#define WHISPER_SERVER_HPP

#include "audio_buffer.hpp"
#include "whisper.h"

#include <string>
#include <vector>
#include <deque>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_map>

// Server configuration
struct ServerConfig {
    std::string model_path = "models/ggml-base.en.bin";
    std::string language = "en";
    std::string host = "0.0.0.0";       // Bind address (all interfaces by default)
    int port = 9090;
    int n_contexts = 2;       // Number of parallel whisper contexts
    int n_threads = 4;        // Threads per inference
    int step_ms = 500;        // Run inference every N ms
    int length_ms = 5000;     // Audio context window
    int keep_ms = 200;        // Overlap between windows
    bool use_gpu = true;
    bool flash_attn = true;
    bool translate = false;

    // VAD configuration (required)
    std::string vad_model_path = "";    // Path to VAD model (required)
    float vad_threshold = 0.5f;
    int vad_check_ms = 30;              // VAD cadence
    int silence_trigger_ms = 1000;      // Silence before final
    int min_speech_ms = 100;            // Ignore short utterances

    // Authentication
    std::string auth_token = "";        // Empty = no auth required

    // Batch transcription (HTTP upload) limits
    size_t max_upload_size_bytes = 25 * 1024 * 1024; // default: 25MB
    int max_audio_duration_ms = 300000;              // default: 5 minutes
    int batch_timeout_ms = 10000;                    // default: 10s
};

struct BatchWordTiming {
    std::string word;
    int start_ms = 0;
    int end_ms = 0;
    float probability = 0.0f;
};

struct BatchTranscribeResponse {
    int status = 200;
    std::string body_json;
};

// Forward declarations
struct Session;
class WhisperServer;

// VAD speech state (managed by inference thread)
enum class SpeechState { IDLE, WAITING_FOR_CONTEXT, SPEAKING, ENDING };

// Context slot in the pool
struct ContextSlot {
    whisper_context* ctx = nullptr;
    bool in_use = false;  // Changed from atomic - we'll protect with mutex
    int slot_id = 0;
};

// Per-connection session
struct Session {
    std::string id;
    std::unique_ptr<AudioBuffer> audio;
    std::vector<float> pcmf32_old;    // Previous audio for overlap
    std::string last_text;             // For detecting changes
    ContextSlot* context_slot = nullptr;
    std::atomic<bool> active{true};
    std::atomic<bool> inference_running{false};

    // VAD state (managed by inference thread)
    SpeechState speech_state = SpeechState::IDLE;
    int64_t speech_start_ms = 0;        // When speech began
    int64_t last_speech_ms = 0;         // Last VAD-positive timestamp
    int64_t waiting_start_ms = 0;       // When we started waiting for context
    std::string pending_text;           // Last partial for potential final

    // WebSocket handle (event loop thread only)
    void* ws_handle = nullptr;

    // Whether a flush has been scheduled (prevents spamming defer)
    std::atomic<bool> flush_pending{false};

    // Thread-safe outgoing message queue
    // Inference thread enqueues messages; uWS event loop thread drains them
    std::mutex outgoing_mutex;
    std::deque<std::string> outgoing_messages;

    void enqueueMessage(const std::string& msg) {
        std::lock_guard<std::mutex> lock(outgoing_mutex);
        outgoing_messages.push_back(msg);
    }

    // Returns all pending messages and clears the queue
    std::deque<std::string> drainMessages() {
        std::lock_guard<std::mutex> lock(outgoing_mutex);
        std::deque<std::string> messages;
        messages.swap(outgoing_messages);
        return messages;
    }
};

// Main server class
class WhisperServer {
public:
    explicit WhisperServer(const ServerConfig& config);
    ~WhisperServer();

    // Initialize contexts and start server
    bool init();

    // Run the server (blocking)
    void run();

    // Stop the server
    void stop();

    // Get server status
    bool isRunning() const { return running_.load(); }

    // === Public methods for WebSocket handlers ===

    // Session management
    std::shared_ptr<Session> createSession(const std::string& id);
    void destroySession(const std::string& id);

    // Get pending messages for a session (called from uWS event loop thread)
    std::deque<std::string> drainSessionMessages(const std::string& session_id);

    // Audio processing
    void onAudioReceived(const std::string& session_id, const int16_t* data, size_t len);

    // Event loop integration for message flushing
    void setEventLoop(void* loop);
    void attachWebSocket(const std::string& session_id, void* ws_handle);
    void detachWebSocket(const std::string& session_id);

    // JSON message helpers
    std::string makeReadyMessage();
    std::string makePartialMessage(const std::string& text);
    std::string makeFinalMessage(const std::string& text);
    std::string makeErrorMessage(const std::string& error);
    BatchTranscribeResponse transcribeWavBytes(const std::vector<uint8_t>& wav_bytes);

private:
    ServerConfig config_;
    std::vector<std::unique_ptr<ContextSlot>> context_pool_;  // Use unique_ptr
    std::mutex context_pool_mutex_;  // Protect context pool access
    std::condition_variable context_pool_cv_;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
    std::mutex sessions_mutex_;

    std::atomic<bool> running_{false};
    std::thread inference_thread_;

    // VAD context (shared across sessions, mutex-protected)
    whisper_vad_context* vad_ctx_ = nullptr;
    std::mutex vad_mutex_;

    // Event loop for deferred message flushing
    void* loop_ = nullptr;

    // Context pool management
    ContextSlot* acquireContext();
    ContextSlot* acquireContextWithTimeout(int timeout_ms);
    void releaseContext(ContextSlot* slot);

    // Inference loop
    void inferenceLoop();
    void runInference(std::shared_ptr<Session> session);

    // VAD methods
    float detectSpeechProb(const float* samples, int n_samples);
    void updateVADState(std::shared_ptr<Session> session, int64_t now_ms);
    void emitFinal(std::shared_ptr<Session> session);

    // Message flush methods
    void notifySessionHasMessages(const std::string& session_id);
    void flushSessionMessagesOnEventLoop(const std::string& session_id);
};

#endif // WHISPER_SERVER_HPP
