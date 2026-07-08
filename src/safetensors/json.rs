// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal zero-dependency JSON parser for Safetensors headers.
//!
//! Safetensors headers are simple JSON objects with:
//! - String keys
//! - String, number (integer), array, and object values
//! - A special `__metadata__` key with string->string mappings
//!
//! This parser implements only what's needed to parse Safetensors headers
//! without pulling in serde_json or other dependencies.

use crate::error::{ParserError, Result};
use std::collections::BTreeMap;

/// Parsed JSON value.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(i64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Get this value as a string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get this value as an integer, if it is one.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get this value as a usize, if it is a non-negative integer.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().and_then(|n| n.try_into().ok())
    }

    /// Get this value as an array, if it is one.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Get this value as an object, if it is one.
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Extract an array of usize values from this value.
    pub fn as_usize_array(&self) -> Option<Vec<usize>> {
        let arr = self.as_array()?;
        arr.iter().map(|v| v.as_usize()).collect()
    }
}

/// Parse a JSON string into a JsonValue.
pub fn parse_json(input: &str, path: &str) -> Result<JsonValue> {
    let mut parser = JsonParser::new(input, path);
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if !parser.is_at_end() {
        return Err(parser.error("trailing content after JSON value"));
    }
    Ok(value)
}

struct JsonParser<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    path: &'a str,
    offset: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str, path: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().peekable(),
            path,
            offset: 0,
        }
    }

    fn error(&self, reason: &str) -> ParserError {
        ParserError::InvalidLayout {
            path: self.path.to_owned(),
            reason: format!("JSON parse error at offset {}: {}", self.offset, reason),
        }
    }

    fn is_at_end(&mut self) -> bool {
        self.chars.peek().is_none()
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
                self.offset += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek_char(&mut self) -> Result<char> {
        self.chars.peek().copied().ok_or_else(|| self.error("unexpected EOF"))
    }

    fn next_char(&mut self) -> Result<char> {
        let c = self.chars.next().ok_or_else(|| self.error("unexpected EOF"))?;
        self.offset += c.len_utf8();
        Ok(c)
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        let c = self.next_char()?;
        if c != expected {
            return Err(self.error(&format!("expected '{}', got '{}'", expected, c)));
        }
        Ok(())
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_whitespace();
        let c = self.peek_char()?;
        match c {
            '"' => self.parse_string().map(JsonValue::String),
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            't' | 'f' => self.parse_bool(),
            'n' => self.parse_null(),
            '-' | '0'..='9' => self.parse_number(),
            _ => Err(self.error(&format!("unexpected character: '{}'", c))),
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect_char('"')?;
        let mut s = String::new();
        loop {
            let c = self.next_char()?;
            match c {
                '"' => return Ok(s),
                '\\' => {
                    let escaped = self.next_char()?;
                    match escaped {
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        '/' => s.push('/'),
                        'b' => s.push('\u{0008}'),
                        'f' => s.push('\u{000C}'),
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        'u' => {
                            let code = self.parse_unicode_escape()?;
                            s.push(char::from_u32(code).ok_or_else(|| self.error("invalid unicode escape"))?);
                        }
                        _ => return Err(self.error(&format!("invalid escape sequence: \\{}", escaped))),
                    }
                }
                c if c < '\u{0020}' => {
                    return Err(self.error("control character in string"));
                }
                _ => s.push(c),
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<u32> {
        let mut code = 0u32;
        for _ in 0..4 {
            let c = self.next_char()?;
            let digit = c.to_digit(16).ok_or_else(|| self.error("invalid hex digit in unicode escape"))?;
            code = code * 16 + digit;
        }
        Ok(code)
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.expect_char('{')?;
        let mut map = BTreeMap::new();
        self.skip_whitespace();
        
        if self.peek_char()? == '}' {
            self.next_char()?;
            return Ok(JsonValue::Object(map));
        }

        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect_char(':')?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_whitespace();
            
            let c = self.peek_char()?;
            if c == ',' {
                self.next_char()?;
            } else if c == '}' {
                self.next_char()?;
                return Ok(JsonValue::Object(map));
            } else {
                return Err(self.error(&format!("expected ',' or '}}' in object, got '{}'", c)));
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.expect_char('[')?;
        let mut arr = Vec::new();
        self.skip_whitespace();
        
        if self.peek_char()? == ']' {
            self.next_char()?;
            return Ok(JsonValue::Array(arr));
        }

        loop {
            let value = self.parse_value()?;
            arr.push(value);
            self.skip_whitespace();
            
            let c = self.peek_char()?;
            if c == ',' {
                self.next_char()?;
            } else if c == ']' {
                self.next_char()?;
                return Ok(JsonValue::Array(arr));
            } else {
                return Err(self.error(&format!("expected ',' or ']' in array, got '{}'", c)));
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let mut s = String::new();
        
        // Optional negative sign
        if self.peek_char()? == '-' {
            s.push(self.next_char()?);
        }

        // Integer part
        let first = self.next_char()?;
        if !first.is_ascii_digit() {
            return Err(self.error("invalid number"));
        }
        s.push(first);

        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() {
                s.push(self.next_char()?);
            } else {
                break;
            }
        }

        // Check for fractional or exponent (we'll reject these for now)
        if let Some(&c) = self.chars.peek() {
            if c == '.' || c == 'e' || c == 'E' {
                return Err(self.error("floating-point numbers not supported in Safetensors headers"));
            }
        }

        let n: i64 = s.parse().map_err(|_| self.error("number out of range"))?;
        Ok(JsonValue::Number(n))
    }

    fn parse_bool(&mut self) -> Result<JsonValue> {
        if self.input[self.offset..].starts_with("true") {
            for _ in 0..4 {
                self.next_char()?;
            }
            Ok(JsonValue::Bool(true))
        } else if self.input[self.offset..].starts_with("false") {
            for _ in 0..5 {
                self.next_char()?;
            }
            Ok(JsonValue::Bool(false))
        } else {
            Err(self.error("invalid boolean"))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue> {
        if self.input[self.offset..].starts_with("null") {
            for _ in 0..4 {
                self.next_char()?;
            }
            Ok(JsonValue::Null)
        } else {
            Err(self.error("invalid null"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_object() {
        let json = r#"{"key": "value", "num": 42}"#;
        let value = parse_json(json, "test").unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj.get("key").unwrap().as_str().unwrap(), "value");
        assert_eq!(obj.get("num").unwrap().as_i64().unwrap(), 42);
    }

    #[test]
    fn parses_arrays() {
        let json = r#"{"shape": [64, 2048], "offsets": [0, 524288]}"#;
        let value = parse_json(json, "test").unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj.get("shape").unwrap().as_usize_array().unwrap(), vec![64, 2048]);
    }

    #[test]
    fn parses_nested_objects() {
        let json = r#"{"__metadata__": {"format": "pt"}}"#;
        let value = parse_json(json, "test").unwrap();
        let obj = value.as_object().unwrap();
        let meta = obj.get("__metadata__").unwrap().as_object().unwrap();
        assert_eq!(meta.get("format").unwrap().as_str().unwrap(), "pt");
    }

    #[test]
    fn parses_escaped_strings() {
        let json = r#"{"key": "value with \"quotes\" and \\backslash"}"#;
        let value = parse_json(json, "test").unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj.get("key").unwrap().as_str().unwrap(), "value with \"quotes\" and \\backslash");
    }

    #[test]
    fn rejects_trailing_content() {
        let json = r#"{"key": "value"} extra"#;
        assert!(parse_json(json, "test").is_err());
    }

    #[test]
    fn rejects_floating_point() {
        let json = r#"{"key": 3.14}"#;
        assert!(parse_json(json, "test").is_err());
    }
}
