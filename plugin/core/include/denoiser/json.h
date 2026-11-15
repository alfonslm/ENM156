#pragma once
// A small JSON reader, enough for the headers of model and vector files
// (plugin/tools/README.md). Not for the audio thread: parsing allocates.
//
//   denoiser::Json h = denoiser::Json::parse(text);   // throws FormatError
//   int n_fft = h.at("derived").at("n_fft").integer();
//   if (const Json* b = h.at("derived").find("mask_bound"); b && !b->isNull()) ...
//
// Numbers are read independently of the C locale (a host may have set one
// with a decimal comma). Nesting is limited to 64 levels, so a hostile file
// cannot overflow the stack.

#include "denoiser/format_error.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace denoiser {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    // Parses one JSON value; only whitespace may follow it. Throws FormatError
    // with the byte position on any syntax error.
    static Json parse(std::string_view text);

    Json() = default;  // null

    Type type() const noexcept { return type_; }
    bool isNull() const noexcept { return type_ == Type::Null; }
    bool isBool() const noexcept { return type_ == Type::Bool; }
    bool isNumber() const noexcept { return type_ == Type::Number; }
    bool isString() const noexcept { return type_ == Type::String; }
    bool isArray() const noexcept { return type_ == Type::Array; }
    bool isObject() const noexcept { return type_ == Type::Object; }

    // Typed access; each throws FormatError if the value has another type.
    bool boolean() const;
    double number() const;
    // A number that is a whole number in [min, max]
    long long integer(long long min = -(1LL << 53), long long max = 1LL << 53) const;
    const std::string& string() const;
    const std::vector<Json>& array() const;
    // Object members in file order
    const std::vector<std::pair<std::string, Json>>& members() const;

    // Object member by key: nullptr if absent (throws if this is not an object)
    const Json* find(std::string_view key) const;
    // Object member by key; throws FormatError naming the key if absent
    const Json& at(std::string_view key) const;

    // "null", "true", "a number", ...: for error messages
    static const char* typeName(Type t) noexcept;

private:
    class Parser;

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Json> array_;
    std::vector<std::pair<std::string, Json>> object_;
};

}  // namespace denoiser
