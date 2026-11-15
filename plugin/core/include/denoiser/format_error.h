#pragma once

#include <stdexcept>
#include <string>

namespace denoiser {

// A model or vectors file (or its JSON header) is malformed, truncated, of an
// unsupported version, or inconsistent. what() says what and where.
class FormatError : public std::runtime_error {
public:
    explicit FormatError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace denoiser
