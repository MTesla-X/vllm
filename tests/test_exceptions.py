# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm/exceptions.py"""

import pytest

from vllm.exceptions import (
    LoRAAdapterNotFoundError,
    VLLMNotFoundError,
    VLLMValidationError,
)

pytestmark = pytest.mark.cpu_test


class TestVLLMValidationError:
    def test_basic_message(self):
        err = VLLMValidationError("something went wrong")
        assert str(err) == "something went wrong"

    def test_message_with_parameter(self):
        err = VLLMValidationError("invalid value", parameter="temperature")
        assert "temperature" in str(err)
        assert "parameter=temperature" in str(err)
        assert err.parameter == "temperature"

    def test_message_with_value(self):
        err = VLLMValidationError("out of range", value=-1.5)
        assert "value=-1.5" in str(err)
        assert err.value == -1.5

    def test_message_with_parameter_and_value(self):
        err = VLLMValidationError("invalid", parameter="top_p", value=2.0)
        result = str(err)
        assert "invalid" in result
        assert "parameter=top_p" in result
        assert "value=2.0" in result

    def test_is_value_error(self):
        err = VLLMValidationError("test")
        assert isinstance(err, ValueError)

    def test_no_extras(self):
        err = VLLMValidationError("plain error")
        # No extras means no parenthetical
        assert str(err) == "plain error"

    def test_raises_as_value_error(self):
        with pytest.raises(ValueError):
            raise VLLMValidationError("test error")


class TestVLLMNotFoundError:
    def test_basic(self):
        err = VLLMNotFoundError("resource not found")
        assert str(err) == "resource not found"
        assert isinstance(err, Exception)


class TestLoRAAdapterNotFoundError:
    def test_message_format(self):
        err = LoRAAdapterNotFoundError(
            lora_name="my_adapter",
            lora_path="/path/to/adapter",
        )
        assert "my_adapter" in str(err)
        assert "/path/to/adapter" in str(err)
        assert "Loading lora" in str(err)
        assert "No adapter found" in str(err)

    def test_is_not_found_error(self):
        err = LoRAAdapterNotFoundError(lora_name="test", lora_path="/tmp/test")
        assert isinstance(err, VLLMNotFoundError)

    def test_message_attribute(self):
        err = LoRAAdapterNotFoundError(
            lora_name="adapter1", lora_path="/models/adapter1"
        )
        assert err.message == str(err)
