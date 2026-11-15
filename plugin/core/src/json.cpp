#include "denoiser/json.h"

#include <cmath>
#include <locale>
#include <sstream>

namespace denoiser {

namespace {
constexpr int kMaxDepth = 64;
}

class Json::Parser {
public:
    explicit Parser(std::string_view text) : s_(text) {}

    Json document() {
        Json v = value(0);
        skipSpace();
        if (pos_ != s_.size())
            fail("unexpected data after the JSON value");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw FormatError("JSON: " + what + " at byte " + std::to_string(pos_));
    }

    void skipSpace() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
            ++pos_;
    }

    char peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }

    void expect(char c) {
        if (pos_ >= s_.size() || s_[pos_] != c)
            fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    void literal(std::string_view word) {
        if (s_.substr(pos_, word.size()) != word)
            fail("invalid literal");
        pos_ += word.size();
    }

    Json value(int depth) {
        if (depth > kMaxDepth)
            fail("nested too deeply");
        skipSpace();
        if (pos_ >= s_.size())
            fail("unexpected end of text");
        Json v;
        switch (s_[pos_]) {
            case '{': v.type_ = Type::Object; object(v, depth); break;
            case '[': v.type_ = Type::Array; array(v, depth); break;
            case '"': v.type_ = Type::String; v.string_ = string(); break;
            case 't': literal("true"); v.type_ = Type::Bool; v.bool_ = true; break;
            case 'f': literal("false"); v.type_ = Type::Bool; v.bool_ = false; break;
            case 'n': literal("null"); break;
            default: v.type_ = Type::Number; v.number_ = number(); break;
        }
        return v;
    }

    void object(Json& v, int depth) {
        expect('{');
        skipSpace();
        if (peek() == '}') {
            ++pos_;
            return;
        }
        for (;;) {
            skipSpace();
            if (peek() != '"')
                fail("expected a string key");
            std::string key = string();
            skipSpace();
            expect(':');
            v.object_.emplace_back(std::move(key), value(depth + 1));
            skipSpace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return;
        }
    }

    void array(Json& v, int depth) {
        expect('[');
        skipSpace();
        if (peek() == ']') {
            ++pos_;
            return;
        }
        for (;;) {
            v.array_.push_back(value(depth + 1));
            skipSpace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return;
        }
    }

    unsigned hex4() {
        if (s_.size() - pos_ < 4)
            fail("truncated \\u escape");
        unsigned code = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[pos_++];
            code <<= 4;
            if (c >= '0' && c <= '9') code |= unsigned(c - '0');
            else if (c >= 'a' && c <= 'f') code |= unsigned(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') code |= unsigned(c - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return code;
    }

    static void appendUtf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (pos_ >= s_.size())
                fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"')
                return out;
            if (static_cast<unsigned char>(c) < 0x20)
                fail("control character in string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size())
                fail("unterminated string");
            switch (s_[pos_++]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {  // high surrogate: a low one must follow
                        if (s_.substr(pos_, 2) != "\\u")
                            fail("unpaired surrogate in \\u escape");
                        pos_ += 2;
                        const unsigned low = hex4();
                        if (low < 0xDC00 || low >= 0xE000)
                            fail("unpaired surrogate in \\u escape");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) {
                        fail("unpaired surrogate in \\u escape");
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: fail("invalid escape in string");
            }
        }
    }

    static bool digit(char c) { return c >= '0' && c <= '9'; }

    double number() {
        // Check the JSON grammar first: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?
        const std::size_t start = pos_;
        if (peek() == '-')
            ++pos_;
        if (peek() == '0') {
            ++pos_;
        } else if (digit(peek())) {
            while (digit(peek())) ++pos_;
        } else {
            fail("invalid value");
        }
        if (peek() == '.') {
            ++pos_;
            if (!digit(peek())) fail("invalid number");
            while (digit(peek())) ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!digit(peek())) fail("invalid number");
            while (digit(peek())) ++pos_;
        }
        // Convert with the classic locale (strtod would follow the C locale)
        std::istringstream in(std::string(s_.substr(start, pos_ - start)));
        in.imbue(std::locale::classic());
        double d = 0.0;
        in >> d;
        if (in.fail() || !std::isfinite(d))
            fail("number out of range");
        return d;
    }

    std::string_view s_;
    std::size_t pos_ = 0;
};

Json Json::parse(std::string_view text) { return Parser(text).document(); }

const char* Json::typeName(Type t) noexcept {
    switch (t) {
        case Type::Null: return "null";
        case Type::Bool: return "a boolean";
        case Type::Number: return "a number";
        case Type::String: return "a string";
        case Type::Array: return "an array";
        case Type::Object: return "an object";
    }
    return "?";
}

namespace {
[[noreturn]] void wrongType(const char* wanted, Json::Type got) {
    throw FormatError(std::string("JSON: expected ") + wanted + ", got " + Json::typeName(got));
}
}  // namespace

bool Json::boolean() const {
    if (type_ != Type::Bool) wrongType("a boolean", type_);
    return bool_;
}

double Json::number() const {
    if (type_ != Type::Number) wrongType("a number", type_);
    return number_;
}

long long Json::integer(long long min, long long max) const {
    const double d = number();
    if (d != std::floor(d) || d < static_cast<double>(min) || d > static_cast<double>(max))
        throw FormatError("JSON: expected a whole number in [" + std::to_string(min) + ", " +
                          std::to_string(max) + "], got " + std::to_string(d));
    return static_cast<long long>(d);
}

const std::string& Json::string() const {
    if (type_ != Type::String) wrongType("a string", type_);
    return string_;
}

const std::vector<Json>& Json::array() const {
    if (type_ != Type::Array) wrongType("an array", type_);
    return array_;
}

const std::vector<std::pair<std::string, Json>>& Json::members() const {
    if (type_ != Type::Object) wrongType("an object", type_);
    return object_;
}

const Json* Json::find(std::string_view key) const {
    for (const auto& [k, v] : members())
        if (k == key)
            return &v;
    return nullptr;
}

const Json& Json::at(std::string_view key) const {
    if (const Json* v = find(key))
        return *v;
    throw FormatError("JSON: missing key \"" + std::string(key) + "\"");
}

}  // namespace denoiser
