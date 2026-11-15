#pragma once
// A minimal WAV (RIFF/WAVE) reader and writer for the command-line tool.
//
// Reads PCM 8 (unsigned), 16, 24 and 32 bit and IEEE float 32 / 64 bit,
// plain or WAVE_FORMAT_EXTENSIBLE, any number of channels; unknown chunks
// are skipped. Not RF64 / files over 4 GB, not compressed formats (A-law,
// ADPCM, ...). Samples become floats in [-1, 1) (PCM: divided by 2^(bits-1)).
//
// Writes PCM 8/16/24/32 bit or float 32/64 bit, interleaved, with a plain
// 16-byte fmt chunk (PCM) or an 18-byte one plus a fact chunk (float).
// PCM is rounded to nearest and clipped to the format's range.
//
// All functions throw wav::Error (a std::runtime_error) with a message.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace wav {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

enum class Format { Pcm8, Pcm16, Pcm24, Pcm32, Float32, Float64 };

const char* formatName(Format f) noexcept;  // "16-bit PCM", "32-bit float", ...

struct Audio {
    int sampleRate = 0;
    Format format = Format::Float32;           // as read (or to write)
    std::vector<std::vector<float>> channels;  // planar: channels[c][frame]

    std::size_t frames() const noexcept { return channels.empty() ? 0 : channels[0].size(); }
};

// Whole file in memory -> Audio. `name` is for messages.
Audio decode(const std::vector<unsigned char>& bytes, const std::string& name);
// Audio -> a whole file. Counts samples clipped (PCM: outside the range,
// or NaN/inf) in *clipped if given.
std::vector<unsigned char> encode(const Audio& audio, Format format, std::size_t* clipped = nullptr);

// Files (std::filesystem::path, so non-ASCII names work on Windows too)
std::vector<unsigned char> readFile(const std::filesystem::path& path);
void writeFile(const std::filesystem::path& path, const std::vector<unsigned char>& bytes);
Audio read(const std::filesystem::path& path);
void write(const std::filesystem::path& path, const Audio& audio, Format format, std::size_t* clipped = nullptr);

// UTF-8 text of a path, for messages
std::string display(const std::filesystem::path& path);

}  // namespace wav
