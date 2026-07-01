# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm/logging_utils/formatter.py"""

import logging
from unittest.mock import patch

import pytest

from vllm.logging_utils.formatter import ColoredFormatter, NewLineFormatter

pytestmark = pytest.mark.cpu_test


# =============================================================================
# NewLineFormatter tests
# =============================================================================


class TestNewLineFormatter:
    @patch("vllm.envs.VLLM_LOGGING_LEVEL", "INFO")
    def test_format_single_line(self):
        fmt = "%(levelname)s %(fileinfo)s:%(lineno)d - %(message)s"
        formatter = NewLineFormatter(fmt)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Hello world",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        assert "Hello world" in result
        assert "INFO" in result

    @patch("vllm.envs.VLLM_LOGGING_LEVEL", "INFO")
    def test_format_uses_filename_when_not_debug(self):
        fmt = "%(levelname)s [%(fileinfo)s:%(lineno)d] %(message)s"
        formatter = NewLineFormatter(fmt)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/home/user/vllm/vllm/engine/core.py",
            lineno=10,
            msg="test message",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        # In non-debug mode, fileinfo should be the filename
        assert "core.py" in result

    @patch("vllm.envs.VLLM_LOGGING_LEVEL", "DEBUG")
    def test_format_uses_relpath_in_debug(self):
        fmt = "%(levelname)s [%(fileinfo)s:%(lineno)d] %(message)s"
        formatter = NewLineFormatter(fmt)
        # Use a path relative to the formatter's root_dir
        root = formatter.root_dir
        pathname = str(root / "vllm" / "engine" / "core.py")
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname=pathname,
            lineno=10,
            msg="debug message",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        # In debug mode, path is shortened
        assert "engine" in result

    @patch("vllm.envs.VLLM_LOGGING_LEVEL", "INFO")
    def test_multiline_message_prefixed(self):
        fmt = "%(levelname)s - %(message)s"
        formatter = NewLineFormatter(fmt)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/test.py",
            lineno=1,
            msg="line1\nline2\nline3",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        # Newlines are replaced with \r\n + prefix
        assert "\r\n" in result


# =============================================================================
# ColoredFormatter tests
# =============================================================================


class TestColoredFormatter:
    def test_format_adds_color_to_levelname(self):
        fmt = "%(levelname)s - %(message)s"
        formatter = ColoredFormatter(fmt)
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="/test.py",
            lineno=1,
            msg="error occurred",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        # Should contain ANSI color codes
        assert "\033[" in result
        assert "error occurred" in result

    def test_format_restores_levelname(self):
        fmt = "%(levelname)s - %(message)s"
        formatter = ColoredFormatter(fmt)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="/test.py",
            lineno=1,
            msg="warning",
            args=None,
            exc_info=None,
        )
        formatter.format(record)
        # Original levelname should be restored
        assert record.levelname == "WARNING"

    def test_all_log_levels_have_colors(self):
        assert "DEBUG" in ColoredFormatter.COLORS
        assert "INFO" in ColoredFormatter.COLORS
        assert "WARNING" in ColoredFormatter.COLORS
        assert "ERROR" in ColoredFormatter.COLORS
        assert "CRITICAL" in ColoredFormatter.COLORS

    def test_grey_color_in_timestamp(self):
        fmt = "%(asctime)s %(levelname)s [%(fileinfo)s:%(lineno)d] %(message)s"
        formatter = ColoredFormatter(fmt)
        # The format string should have been modified to include grey
        assert ColoredFormatter.GREY in formatter._fmt

    def test_format_info_level(self):
        fmt = "%(levelname)s %(message)s"
        formatter = ColoredFormatter(fmt)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/test.py",
            lineno=1,
            msg="info message",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        # Green color for INFO
        assert "\033[32m" in result
        assert "info message" in result
