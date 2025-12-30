#pragma once

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

enum class LogLevel {
    Info,
    Debug,
};

static inline LogLevel logLevelFromEnv() {
    static LogLevel level = []() -> LogLevel {
        const char* env = std::getenv("INFO_LEVEL");
        if (!env || env[0] == '\0') {
            return LogLevel::Info;
        }
        std::string value(env);
        for (char& c : value) {
            c = (char)std::tolower((unsigned char)c);
        }
        if (value == "debug") {
            return LogLevel::Debug;
        }
        return LogLevel::Info;
    }();

    return level;
}

static inline void logInfo(const std::string& msg) {
    std::cout << "[whisper-server] " << msg << std::endl;
}

static inline void logDebug(const std::string& msg) {
    if (logLevelFromEnv() == LogLevel::Debug) {
        std::cout << "[whisper-server][debug] " << msg << std::endl;
    }
}
