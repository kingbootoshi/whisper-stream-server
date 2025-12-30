#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

struct WavPcm16Mono {
    int sample_rate = 0;
    std::vector<float> samples;
};

static inline uint16_t read_le16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t read_le32(const uint8_t* p) {
    return (uint32_t)p[0]
        | ((uint32_t)p[1] << 8)
        | ((uint32_t)p[2] << 16)
        | ((uint32_t)p[3] << 24);
}

static inline bool starts_with_4(const uint8_t* p, const char* s4) {
    return p[0] == (uint8_t)s4[0] && p[1] == (uint8_t)s4[1] && p[2] == (uint8_t)s4[2] && p[3] == (uint8_t)s4[3];
}

static inline bool parse_wav_pcm16_mono_from_bytes(
    const uint8_t* data,
    size_t size,
    WavPcm16Mono& out,
    std::string& err
) {
    out = WavPcm16Mono{};
    err.clear();

    if (!data || size < 12) {
        err = "WAV too small (missing RIFF header)";
        return false;
    }

    if (!starts_with_4(data, "RIFF")) {
        err = "Not RIFF (expected 'RIFF')";
        return false;
    }
    if (!starts_with_4(data + 8, "WAVE")) {
        err = "Not WAVE (expected 'WAVE')";
        return false;
    }

    bool found_fmt = false;
    bool found_data = false;

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;

    const uint8_t* data_chunk = nullptr;
    uint32_t data_chunk_size = 0;

    size_t off = 12;
    while (off + 8 <= size) {
        const uint8_t* chunk_id = data + off;
        uint32_t chunk_size = read_le32(data + off + 4);
        off += 8;

        if (off + chunk_size > size) {
            err = "Invalid chunk size (out of bounds)";
            return false;
        }

        if (starts_with_4(chunk_id, "fmt ")) {
            if (chunk_size < 16) {
                err = "fmt chunk too small";
                return false;
            }
            audio_format = read_le16(data + off + 0);
            num_channels = read_le16(data + off + 2);
            sample_rate = read_le32(data + off + 4);
            bits_per_sample = read_le16(data + off + 14);
            found_fmt = true;
        } else if (starts_with_4(chunk_id, "data")) {
            data_chunk = data + off;
            data_chunk_size = chunk_size;
            found_data = true;
        }

        off += chunk_size;
        if (chunk_size & 1) off += 1;
        if (off > size) break;
    }

    if (!found_fmt) {
        err = "Missing fmt chunk";
        return false;
    }
    if (!found_data) {
        err = "Missing data chunk";
        return false;
    }

    if (audio_format != 1) {
        err = "Unsupported WAV format (only PCM=1 supported)";
        return false;
    }
    if (num_channels != 1) {
        err = "Unsupported WAV channels (only mono supported)";
        return false;
    }
    if (bits_per_sample != 16) {
        err = "Unsupported WAV bit depth (only 16-bit supported)";
        return false;
    }
    if (sample_rate == 0) {
        err = "Invalid sample rate";
        return false;
    }

    if (data_chunk_size % 2 != 0) {
        data_chunk_size -= 1;
    }

    const size_t n_samples = data_chunk_size / 2;
    out.sample_rate = (int)sample_rate;
    out.samples.resize(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        const uint8_t* p = data_chunk + i * 2;
        int16_t s = (int16_t)read_le16(p);
        out.samples[i] = (float)s / 32768.0f;
    }

    return true;
}

static inline std::vector<float> resample_linear(
    const std::vector<float>& input,
    int from_rate,
    int to_rate
) {
    if (from_rate == to_rate || input.empty()) {
        return input;
    }

    const double ratio = (double)from_rate / (double)to_rate;
    const size_t out_len = (size_t)std::llround((double)input.size() / ratio);
    std::vector<float> out(out_len);

    for (size_t i = 0; i < out_len; ++i) {
        const double src = (double)i * ratio;
        const size_t i0 = (size_t)std::floor(src);
        const size_t i1 = std::min(i0 + 1, input.size() - 1);
        const double t = src - (double)i0;
        out[i] = (float)((1.0 - t) * input[i0] + t * input[i1]);
    }

    return out;
}
