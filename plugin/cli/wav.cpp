#include "wav.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace wav {

namespace {

std::uint32_t u16(const unsigned char* p) { return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8; }
std::uint32_t u32(const unsigned char* p) { return u16(p) | u16(p + 2) << 16; }

void put16(std::vector<unsigned char>& out, std::uint32_t v) {
    out.push_back(static_cast<unsigned char>(v & 0xff));
    out.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
}
void put32(std::vector<unsigned char>& out, std::uint32_t v) {
    put16(out, v & 0xffff);
    put16(out, v >> 16);
}
void putId(std::vector<unsigned char>& out, const char* id) { out.insert(out.end(), id, id + 4); }

int bitsOf(Format f) {
    switch (f) {
        case Format::Pcm8: return 8;
        case Format::Pcm16: return 16;
        case Format::Pcm24: return 24;
        case Format::Pcm32: return 32;
        case Format::Float32: return 32;
        case Format::Float64: return 64;
    }
    return 0;
}

bool isFloat(Format f) { return f == Format::Float32 || f == Format::Float64; }

// The 14 bytes after the format code in a WAVE_FORMAT_EXTENSIBLE sub-format GUID
const unsigned char kGuidTail[14] = {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};

}  // namespace

const char* formatName(Format f) noexcept {
    switch (f) {
        case Format::Pcm8: return "8-bit PCM";
        case Format::Pcm16: return "16-bit PCM";
        case Format::Pcm24: return "24-bit PCM";
        case Format::Pcm32: return "32-bit PCM";
        case Format::Float32: return "32-bit float";
        case Format::Float64: return "64-bit float";
    }
    return "?";
}

Audio decode(const std::vector<unsigned char>& bytes, const std::string& name) {
    const std::size_t size = bytes.size();
    const unsigned char* b = bytes.data();
    if (size < 12 || std::memcmp(b, "RIFF", 4) != 0 || std::memcmp(b + 8, "WAVE", 4) != 0) {
        if (size >= 4 && (std::memcmp(b, "RF64", 4) == 0 || std::memcmp(b, "RIFX", 4) == 0))
            throw Error(name + ": RF64 / big-endian RIFX WAV files are not supported");
        throw Error(name + ": not a WAV file (no RIFF/WAVE header)");
    }
    bool haveFmt = false;
    Format format = Format::Pcm16;
    int channels = 0, rate = 0, blockAlign = 0;
    const unsigned char* data = nullptr;
    std::size_t dataSize = 0;
    std::size_t pos = 12;
    while (pos + 8 <= size) {
        const unsigned char* id = b + pos;
        const std::size_t chunk = u32(b + pos + 4);
        const std::size_t body = pos + 8;
        const std::size_t avail = size - body;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (chunk < 16 || chunk > avail)
                throw Error(name + ": malformed fmt chunk");
            const unsigned char* f = b + body;
            std::uint32_t code = u16(f);
            channels = int(u16(f + 2));
            const std::uint32_t sr = u32(f + 4);
            blockAlign = int(u16(f + 12));
            const int bits = int(u16(f + 14));
            if (code == 0xFFFE) {
                if (chunk < 40)
                    throw Error(name + ": malformed WAVE_FORMAT_EXTENSIBLE fmt chunk");
                if (std::memcmp(f + 26, kGuidTail, sizeof kGuidTail) != 0)
                    throw Error(name + ": unsupported WAVE_FORMAT_EXTENSIBLE sub-format");
                code = u16(f + 24);
            }
            if (code == 1 && bits == 8) format = Format::Pcm8;
            else if (code == 1 && bits == 16) format = Format::Pcm16;
            else if (code == 1 && bits == 24) format = Format::Pcm24;
            else if (code == 1 && bits == 32) format = Format::Pcm32;
            else if (code == 3 && bits == 32) format = Format::Float32;
            else if (code == 3 && bits == 64) format = Format::Float64;
            else
                throw Error(name + ": unsupported WAV encoding (format code " + std::to_string(code) + ", " +
                            std::to_string(bits) + " bits); supported: PCM 8/16/24/32 bit, float 32/64 bit");
            if (channels < 1)
                throw Error(name + ": no channels");
            if (sr < 1 || sr > 10000000)
                throw Error(name + ": bad sample rate " + std::to_string(sr));
            rate = int(sr);
            if (blockAlign != channels * bits / 8)
                throw Error(name + ": inconsistent block size in the fmt chunk");
            haveFmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = b + body;
            // Writers that stream may leave the size at 0 / 0xFFFFFFFF: take what is there
            dataSize = chunk > avail || chunk == 0 ? avail : chunk;
            if (haveFmt)
                break;
        }
        if (chunk > avail)
            break;
        pos = body + chunk + (chunk & 1);  // chunks are padded to even sizes
    }
    if (!haveFmt)
        throw Error(name + ": no fmt chunk");
    if (!data)
        throw Error(name + ": no data chunk");

    Audio a;
    a.sampleRate = rate;
    a.format = format;
    const std::size_t frames = dataSize / std::size_t(blockAlign);
    const std::size_t width = std::size_t(bitsOf(format) / 8);
    a.channels.assign(std::size_t(channels), std::vector<float>(frames));
    for (std::size_t t = 0; t < frames; ++t) {
        for (std::size_t c = 0; c < std::size_t(channels); ++c) {
            const unsigned char* p = data + t * std::size_t(blockAlign) + c * width;
            float v = 0.0f;
            switch (format) {
                case Format::Pcm8: v = float(int(p[0]) - 128) / 128.0f; break;
                case Format::Pcm16: v = float(std::int16_t(std::uint16_t(u16(p)))) / 32768.0f; break;
                case Format::Pcm24: {
                    std::int32_t s = std::int32_t(std::uint32_t(p[0]) << 8 | std::uint32_t(p[1]) << 16 | std::uint32_t(p[2]) << 24);
                    v = float(double(s / 256) / 8388608.0);
                    break;
                }
                case Format::Pcm32: v = float(double(std::int32_t(u32(p))) / 2147483648.0); break;
                case Format::Float32: {
                    const std::uint32_t u = u32(p);
                    std::memcpy(&v, &u, 4);
                    break;
                }
                case Format::Float64: {
                    const std::uint64_t u = std::uint64_t(u32(p)) | std::uint64_t(u32(p + 4)) << 32;
                    double d;
                    std::memcpy(&d, &u, 8);
                    v = float(d);
                    break;
                }
            }
            a.channels[c][t] = v;
        }
    }
    return a;
}

std::vector<unsigned char> encode(const Audio& audio, Format format, std::size_t* clipped) {
    const std::size_t channels = audio.channels.size();
    const std::size_t frames = audio.frames();
    for (const auto& ch : audio.channels)
        if (ch.size() != frames)
            throw Error("channels of different lengths");
    if (channels < 1 || channels > 65535)
        throw Error("bad number of channels");
    if (audio.sampleRate < 1)
        throw Error("bad sample rate");
    const int bits = bitsOf(format);
    const std::size_t width = std::size_t(bits / 8);
    const std::uint64_t dataSize = std::uint64_t(frames) * channels * width;
    const bool flt = isFloat(format);
    const std::uint64_t fmtSize = flt ? 18 : 16;
    const std::uint64_t riffSize = 4 + (8 + fmtSize) + (flt ? 12 : 0) + 8 + dataSize + (dataSize & 1);
    if (riffSize > 0xFFFFFFFFull)
        throw Error("the output would be larger than a WAV file can be (4 GB)");

    std::vector<unsigned char> out;
    out.reserve(std::size_t(riffSize + 8));
    putId(out, "RIFF");
    put32(out, std::uint32_t(riffSize));
    putId(out, "WAVE");
    putId(out, "fmt ");
    put32(out, std::uint32_t(fmtSize));
    put16(out, flt ? 3 : 1);
    put16(out, std::uint32_t(channels));
    put32(out, std::uint32_t(audio.sampleRate));
    put32(out, std::uint32_t(std::uint64_t(audio.sampleRate) * channels * width));
    put16(out, std::uint32_t(channels * width));
    put16(out, std::uint32_t(bits));
    if (flt) {
        put16(out, 0);  // cbSize
        putId(out, "fact");
        put32(out, 4);
        put32(out, std::uint32_t(frames));
    }
    putId(out, "data");
    put32(out, std::uint32_t(dataSize));

    std::size_t clips = 0;
    // PCM: round to nearest, clip to [-2^(bits-1), 2^(bits-1) - 1]
    auto quantise = [&](float x, double scale) -> std::int64_t {
        if (!std::isfinite(x)) {
            ++clips;
            return 0;
        }
        double v = std::nearbyint(double(x) * scale);
        if (v > scale - 1) {
            ++clips;
            v = scale - 1;
        } else if (v < -scale) {
            ++clips;
            v = -scale;
        }
        return std::int64_t(v);
    };
    for (std::size_t t = 0; t < frames; ++t) {
        for (std::size_t c = 0; c < channels; ++c) {
            const float x = audio.channels[c][t];
            switch (format) {
                case Format::Pcm8: out.push_back(static_cast<unsigned char>(quantise(x, 128.0) + 128)); break;
                case Format::Pcm16: put16(out, std::uint32_t(quantise(x, 32768.0)) & 0xffff); break;
                case Format::Pcm24: {
                    const auto v = std::uint32_t(quantise(x, 8388608.0));
                    out.push_back(static_cast<unsigned char>(v & 0xff));
                    out.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
                    out.push_back(static_cast<unsigned char>((v >> 16) & 0xff));
                    break;
                }
                case Format::Pcm32: put32(out, std::uint32_t(quantise(x, 2147483648.0))); break;
                case Format::Float32: {
                    std::uint32_t u;
                    std::memcpy(&u, &x, 4);
                    put32(out, u);
                    break;
                }
                case Format::Float64: {
                    const double d = x;
                    std::uint64_t u;
                    std::memcpy(&u, &d, 8);
                    put32(out, std::uint32_t(u & 0xffffffffu));
                    put32(out, std::uint32_t(u >> 32));
                    break;
                }
            }
        }
    }
    if (dataSize & 1)
        out.push_back(0);
    if (clipped)
        *clipped = clips;
    return out;
}

std::string display(const std::filesystem::path& path) {
    const auto u8 = path.u8string();  // std::string in C++17, std::u8string in C++20
    return std::string(u8.begin(), u8.end());
}

std::vector<unsigned char> readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw Error("can't open " + display(path) + " for reading");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad())
        throw Error("error reading " + display(path));
    return bytes;
}

void writeFile(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw Error("can't open " + display(path) + " for writing");
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out)
        throw Error("error writing " + display(path));
}

Audio read(const std::filesystem::path& path) { return decode(readFile(path), display(path)); }

void write(const std::filesystem::path& path, const Audio& audio, Format format, std::size_t* clipped) {
    writeFile(path, encode(audio, format, clipped));
}

}  // namespace wav
