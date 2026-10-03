#pragma once
// ── mini_json ──────────────────────────────────────────────
// 標準ライブラリのみの最小 JSON パーサ（チューニング設定の読み込み用）。
// 対応: object / array / string / number / true / false / null
// 文字列は \" \\ \/ \b \f \n \r \t \uXXXX(BMP) をデコードする。
// 解析エラーは std::runtime_error（位置つき）を投げる。

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace myln {
namespace json {

struct Value;
using Array  = std::vector<Value>;
using Object = std::map<std::string, Value>;

struct Value {
    enum Type { Null, Bool, Number, String, Arr, Obj } type = Null;
    bool        b = false;
    double      n = 0.0;
    std::string s;
    std::shared_ptr<Array>  a;
    std::shared_ptr<Object> o;

    bool is_null()   const { return type == Null;   }
    bool is_bool()   const { return type == Bool;   }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array()  const { return type == Arr;    }
    bool is_object() const { return type == Obj;    }

    const Array&  arr() const { return *a; }
    const Object& obj() const { return *o; }

    // object のキー参照（無ければ nullptr）
    const Value* find(const std::string& k) const {
        if (type != Obj) return nullptr;
        auto it = o->find(k);
        return it == o->end() ? nullptr : &it->second;
    }
};

class Parser {
    const std::string& t_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(i_) + ": " + msg);
    }
    void ws() {
        while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\t' || t_[i_] == '\n' || t_[i_] == '\r')) ++i_;
    }
    bool eat(char c) { ws(); if (i_ < t_.size() && t_[i_] == c) { ++i_; return true; } return false; }
    void expect(char c) { if (!eat(c)) fail(std::string("expected '") + c + "'"); }

    static void put_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    }

    std::string parse_string_raw() {
        expect('"');
        std::string out;
        while (true) {
            if (i_ >= t_.size()) fail("unterminated string");
            char c = t_[i_++];
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            if (i_ >= t_.size()) fail("bad escape");
            char e = t_[i_++];
            switch (e) {
                case '"': out += '"'; break;   case '\\': out += '\\'; break;
                case '/': out += '/'; break;   case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;  case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;  case 't': out += '\t'; break;
                case 'u': {
                    if (i_ + 4 > t_.size()) fail("bad \\u escape");
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = t_[i_++]; cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= h - '0';
                        else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                        else fail("bad hex digit in \\u escape");
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("unknown escape");
            }
        }
        return out;
    }

    Value parse_value(int depth) {
        if (depth > 64) fail("nesting too deep");
        ws();
        if (i_ >= t_.size()) fail("unexpected end");
        char c = t_[i_];
        Value v;
        if (c == '{') {
            ++i_; v.type = Value::Obj; v.o = std::make_shared<Object>();
            if (eat('}')) return v;
            while (true) {
                ws();
                std::string k = parse_string_raw();
                expect(':');
                (*v.o)[k] = parse_value(depth + 1);
                if (eat(',')) continue;
                expect('}'); break;
            }
        } else if (c == '[') {
            ++i_; v.type = Value::Arr; v.a = std::make_shared<Array>();
            if (eat(']')) return v;
            while (true) {
                v.a->push_back(parse_value(depth + 1));
                if (eat(',')) continue;
                expect(']'); break;
            }
        } else if (c == '"') {
            v.type = Value::String; v.s = parse_string_raw();
        } else if (t_.compare(i_, 4, "true") == 0)  { i_ += 4; v.type = Value::Bool; v.b = true; }
        else if (t_.compare(i_, 5, "false") == 0)   { i_ += 5; v.type = Value::Bool; v.b = false; }
        else if (t_.compare(i_, 4, "null") == 0)    { i_ += 4; v.type = Value::Null; }
        else {
            const char* start = t_.c_str() + i_;
            char* end = nullptr;
            double d = std::strtod(start, &end);
            if (end == start) fail("unexpected character");
            i_ += (size_t)(end - start);
            v.type = Value::Number; v.n = d;
        }
        return v;
    }

public:
    explicit Parser(const std::string& text) : t_(text) {}
    Value parse() {
        Value v = parse_value(0);
        ws();
        if (i_ != t_.size()) fail("trailing characters");
        return v;
    }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

} // namespace json
} // namespace myln
