import pytest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.features.text_preprocessor import (
    clean_text,
    remove_urls,
    remove_punctuation,
    remove_numbers,
    remove_stopwords,
    lemmatize_text,
    remove_extra_spaces
)


class TestRemoveUrls:
    def test_removes_http_url(self):
        assert remove_urls("visit http://example.com today") == "visit  today"

    def test_removes_https_url(self):
        assert remove_urls("go to https://example.com now") == "go to  now"

    def test_no_url(self):
        assert remove_urls("hello world") == "hello world"


class TestRemovePunctuation:
    def test_removes_punctuation(self):
        assert remove_punctuation("hello!!!") == "hello"

    def test_removes_comma(self):
        assert remove_punctuation("hello, world") == "hello world"

    def test_no_punctuation(self):
        assert remove_punctuation("hello world") == "hello world"


class TestRemoveNumbers:
    def test_removes_numbers(self):
        assert remove_numbers("day123") == "day"

    def test_no_numbers(self):
        assert remove_numbers("hello world") == "hello world"

    def test_only_numbers(self):
        assert remove_numbers("12345") == ""


class TestRemoveExtraSpaces:
    def test_removes_extra_spaces(self):
        assert remove_extra_spaces("hello   world") == "hello world"

    def test_leading_trailing_spaces(self):
        assert remove_extra_spaces("  hello  ") == "hello"

    def test_no_extra_spaces(self):
        assert remove_extra_spaces("hello world") == "hello world"


class TestCleanText:
    def test_full_pipeline(self):
        text = "I visited http://example.com 123 times!!!"
        result = clean_text(text)
        assert "http" not in result
        assert "123" not in result
        assert "!!!" not in result

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_returns_string(self):
        result = clean_text("Some sample text")
        assert isinstance(result, str)

    def test_lowercase(self):
        result = clean_text("HELLO WORLD")
        assert result == result.lower()