#include "whisper_server.hpp"
#include "json.hpp"
#include "logging.hpp"

#include <App.h>  // uWebSockets

#include <iostream>
#include <string>
#include <string_view>
#include <csignal>
#include <cstring>
#include <memory>
#include <atomic>
#include <vector>
#include <thread>
#include <cstdlib>
#include <cerrno>

using json = nlohmann::json;

// Global pointers for signal handling
static WhisperServer* g_server = nullptr;
static us_listen_socket_t* g_listen_socket = nullptr;
static uWS::Loop* g_loop = nullptr;

void signalHandler(int signum) {
    std::cout << "\n[whisper-server] Received signal " << signum << ", shutting down..." << std::endl;

    // Close the listen socket to stop accepting new connections
    // and cause the event loop to exit
    if (g_loop && g_listen_socket) {
        g_loop->defer([]{
            if (g_listen_socket) {
                us_listen_socket_close(0, g_listen_socket);
                g_listen_socket = nullptr;
            }
        });
    }

    if (g_server) {
        g_server->stop();
    }
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Required:\n"
              << "  -m, --model PATH      Path to whisper model\n"
              << "      --vad-model PATH  Path to VAD model\n"
              << "\nOptions:\n"
              << "  -p, --port PORT       Port to listen on (default: 9090)\n"
              << "      --host ADDRESS    Bind address (default: 0.0.0.0)\n"
              << "      --token SECRET    Authentication token for WebSocket connections\n"
              << "  -c, --contexts N      Number of parallel contexts (default: 2)\n"
              << "  -t, --threads N       Threads per inference (default: 4)\n"
              << "  -l, --language LANG   Language code (default: en)\n"
              << "      --step MS         Inference step interval in ms (default: 500)\n"
              << "      --length MS       Audio context length in ms (default: 5000)\n"
              << "      --keep MS         Audio overlap in ms (default: 200)\n"
              << "      --no-gpu          Disable GPU acceleration\n"
              << "      --translate       Translate to English\n"
              << "      --vad-threshold N Speech probability threshold 0.0-1.0 (default: 0.5)\n"
              << "      --vad-silence MS  Silence duration to trigger final (default: 1000)\n"
              << "      --max-upload-size MB     Max upload size in MB (default: 25)\n"
              << "      --max-audio-duration MS  Max audio duration in ms (default: 300000)\n"
              << "      --batch-timeout MS       Context acquire timeout for batch (default: 10000)\n"
              << "  -h, --help            Show this help\n"
              << std::endl;
}

bool parseArgs(int argc, char** argv, ServerConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.model_path = argv[++i];
        }
        else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        }
        else if ((arg == "-c" || arg == "--contexts") && i + 1 < argc) {
            config.n_contexts = std::stoi(argv[++i]);
        }
        else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            config.n_threads = std::stoi(argv[++i]);
        }
        else if ((arg == "-l" || arg == "--language") && i + 1 < argc) {
            config.language = argv[++i];
        }
        else if (arg == "--step" && i + 1 < argc) {
            config.step_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--length" && i + 1 < argc) {
            config.length_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--keep" && i + 1 < argc) {
            config.keep_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--no-gpu") {
            config.use_gpu = false;
        }
        else if (arg == "--translate") {
            config.translate = true;
        }
        else if (arg == "--vad-model" && i + 1 < argc) {
            config.vad_model_path = argv[++i];
        }
        else if (arg == "--vad-threshold" && i + 1 < argc) {
            config.vad_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--vad-silence" && i + 1 < argc) {
            config.silence_trigger_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        }
        else if (arg == "--token" && i + 1 < argc) {
            config.auth_token = argv[++i];
        }
        else if (arg == "--max-upload-size" && i + 1 < argc) {
            int mb = std::stoi(argv[++i]);
            if (mb <= 0) {
                std::cerr << "Invalid --max-upload-size (must be > 0)\n";
                return false;
            }
            config.max_upload_size_bytes = (size_t)mb * 1024ull * 1024ull;
        }
        else if (arg == "--max-audio-duration" && i + 1 < argc) {
            config.max_audio_duration_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--batch-timeout" && i + 1 < argc) {
            config.batch_timeout_ms = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }

    // Validate required arguments
    if (config.vad_model_path.empty()) {
        std::cerr << "Error: --vad-model is required\n" << std::endl;
        printUsage(argv[0]);
        return false;
    }

    return true;
}

// Helper to extract query parameter from URL query string
std::string getQueryParam(std::string_view query, const std::string& param) {
    std::string search = param + "=";
    size_t pos = query.find(search);
    if (pos == std::string_view::npos) return "";
    pos += search.length();
    size_t end = query.find('&', pos);
    if (end == std::string_view::npos) end = query.length();
    return std::string(query.substr(pos, end - pos));
}

static bool parseContentLength(std::string_view value, unsigned long long& out) {
    if (value.empty()) return false;
    std::string tmp(value);
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(tmp.c_str(), &end, 10);
    if (errno != 0 || end == tmp.c_str() || *end != '\0') {
        return false;
    }
    out = v;
    return true;
}

static const char* httpStatusLine(int code) {
    switch (code) {
        case 200: return "200 OK";
        case 401: return "401 Unauthorized";
        case 413: return "413 Payload Too Large";
        case 415: return "415 Unsupported Media Type";
        case 422: return "422 Unprocessable Entity";
        case 503: return "503 Service Unavailable";
        default:  return "500 Internal Server Error";
    }
}

static void endJson(uWS::HttpResponse<false>* res, int status, const json& body) {
    res->writeStatus(httpStatusLine(status));
    res->writeHeader("content-type", "application/json");
    res->end(body.dump());
}

// Per-socket user data
struct PerSocketData {
    std::string session_id;
};

int main(int argc, char** argv) {
    ServerConfig config;

    if (!parseArgs(argc, argv, config)) {
        return 1;
    }

    // Create and initialize server
    WhisperServer server(config);
    g_server = &server;

    // Set up signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    if (!server.init()) {
        std::cerr << "[whisper-server] Failed to initialize server" << std::endl;
        return 1;
    }

    // Start inference thread
    server.run();

    // Create uWebSockets app
    uWS::App app;

    app.post("/v1/transcribe", [&server, &config](auto* res, auto* req) {
        if (!config.auth_token.empty()) {
            std::string_view auth = req->getHeader("authorization");
            std::string expected = "Bearer " + config.auth_token;
            if (auth != expected) {
                json j;
                j["error"] = "Unauthorized";
                j["message"] = "Missing/invalid Authorization header";
                endJson(res, 401, j);
                return;
            }
        }

        std::string_view ct = req->getHeader("content-type");
        bool ct_ok =
            ct.rfind("audio/wav", 0) == 0 ||
            ct.rfind("audio/wave", 0) == 0 ||
            ct.rfind("audio/x-wav", 0) == 0;

        if (!ct_ok) {
            json j;
            j["error"] = "Unsupported Media Type";
            j["message"] = "Content-Type must be audio/wav";
            endJson(res, 415, j);
            return;
        }

        std::string_view cl = req->getHeader("content-length");
        unsigned long long content_len = 0;
        if (parseContentLength(cl, content_len)) {
            if (content_len > (unsigned long long)config.max_upload_size_bytes) {
                json j;
                j["error"] = "Payload Too Large";
                j["message"] = "Upload exceeds max_upload_size";
                endJson(res, 413, j);
                return;
            }
        }

        struct UploadState {
            std::vector<uint8_t> body;
            std::atomic<bool> aborted{false};
            std::atomic<bool> responded{false};
            std::atomic<bool> started{false};
            uWS::HttpResponse<false>* res = nullptr;
            uWS::Loop* loop = nullptr;
        };

        auto state = std::make_shared<UploadState>();
        state->res = res;
        state->loop = uWS::Loop::get();

        res->onAborted([state]() {
            state->aborted.store(true);
        });

        res->onData([state, &server, &config](std::string_view chunk, bool isLast) {
            if (state->aborted.load() || state->responded.load()) return;

            if (state->body.size() + chunk.size() > config.max_upload_size_bytes) {
                if (!state->responded.exchange(true)) {
                    json j;
                    j["error"] = "Payload Too Large";
                    j["message"] = "Upload exceeds max_upload_size";
                    endJson(state->res, 413, j);
                }
                return;
            }

            state->body.insert(state->body.end(), chunk.begin(), chunk.end());

            if (!isLast) return;
            if (state->started.exchange(true)) return;

            std::vector<uint8_t> bytes;
            bytes.swap(state->body);

            std::thread([state, bytes = std::move(bytes), &server]() mutable {
                logDebug("Batch upload complete: bytes=" + std::to_string(bytes.size()));
                BatchTranscribeResponse r = server.transcribeWavBytes(bytes);

                if (state->aborted.load()) return;

                state->loop->defer([state, r = std::move(r)]() mutable {
                    if (state->aborted.load() || state->responded.load()) return;
                    state->responded.store(true);
                    state->res->writeStatus(httpStatusLine(r.status));
                    state->res->writeHeader("content-type", "application/json");
                    state->res->end(r.body_json);
                });
            }).detach();
        });
    });

    app
        .ws<PerSocketData>("/*", {
            // Settings
            .compression = uWS::DISABLED,
            .maxPayloadLength = 16 * 1024 * 1024,  // 16MB max message
            .idleTimeout = 120,
            .maxBackpressure = 1 * 1024 * 1024,    // 1MB backpressure

            // Handlers
            .upgrade = [&config](auto* res, auto* req, auto* context) {
                // Check auth token if configured
                if (!config.auth_token.empty()) {
                    std::string token = getQueryParam(req->getQuery(), "token");
                    if (token != config.auth_token) {
                        res->writeStatus("401 Unauthorized");
                        res->end("Invalid or missing token");
                        return;
                    }
                }

                // Accept the upgrade
                res->template upgrade<PerSocketData>(
                    { .session_id = "" },
                    req->getHeader("sec-websocket-key"),
                    req->getHeader("sec-websocket-protocol"),
                    req->getHeader("sec-websocket-extensions"),
                    context
                );
            },

            .open = [&server](auto* ws) {
                // Generate session ID
                static int session_counter = 0;
                std::string session_id = "session_" + std::to_string(++session_counter);

                auto* data = ws->getUserData();
                data->session_id = session_id;

                std::cout << "[whisper-server] WebSocket connected: " << session_id << std::endl;

                // Create the session (no send callback - we use message queue now)
                auto session = server.createSession(session_id);

                if (!session) {
                    ws->send(R"({"type":"error","message":"No available contexts, try again later"})",
                             uWS::OpCode::TEXT);
                    ws->close();
                    return;
                }

                // Attach WebSocket handle for event-driven message flushing
                server.attachWebSocket(session_id, static_cast<void*>(ws));

                // Send ready message (safe: we're on the uWS event loop thread)
                ws->send(server.makeReadyMessage(), uWS::OpCode::TEXT);
            },

            .message = [&server](auto* ws, std::string_view message, uWS::OpCode opCode) {
                auto* data = ws->getUserData();

                if (opCode == uWS::OpCode::BINARY) {
                    // Binary message = audio data (int16 PCM)
                    const int16_t* audio_data = reinterpret_cast<const int16_t*>(message.data());
                    size_t sample_count = message.size() / sizeof(int16_t);

                    server.onAudioReceived(data->session_id, audio_data, sample_count);
                }
                else if (opCode == uWS::OpCode::TEXT) {
                    // Text message = control command (JSON)
                    // For now, we don't handle any text commands
                    std::cout << "[whisper-server] Received text message: " << message << std::endl;
                }
                // Messages are now flushed via event-driven callback (notifySessionHasMessages)
            },

            .close = [&server](auto* ws, int code, std::string_view message) {
                auto* data = ws->getUserData();
                std::cout << "[whisper-server] WebSocket disconnected: " << data->session_id
                          << " (code=" << code << ")" << std::endl;

                // Detach WebSocket before destroying session
                server.detachWebSocket(data->session_id);
                server.destroySession(data->session_id);
            }
        })
        .listen(config.host, config.port, [&config, &server](auto* listen_socket) {
            if (listen_socket) {
                g_listen_socket = listen_socket;
                g_loop = uWS::Loop::get();
                server.setEventLoop(static_cast<void*>(g_loop));
                std::cout << "[whisper-server] Listening on " << config.host << ":" << config.port << std::endl;
                if (!config.auth_token.empty()) {
                    std::cout << "[whisper-server] Token authentication enabled" << std::endl;
                }
            } else {
                std::cerr << "[whisper-server] Failed to listen on " << config.host << ":" << config.port << std::endl;
            }
        })
        .run();

    std::cout << "[whisper-server] Server stopped" << std::endl;
    return 0;
}
