# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm/connections.py"""

import asyncio
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
import requests

from vllm.connections import (
    HTTPConnection,
    _is_retryable,
    global_http_connection,
)

pytestmark = pytest.mark.cpu_test


# =============================================================================
# _is_retryable tests
# =============================================================================


class TestIsRetryable:
    def test_timeout_error_is_retryable(self):
        assert _is_retryable(TimeoutError()) is True

    def test_asyncio_timeout_is_retryable(self):
        assert _is_retryable(asyncio.TimeoutError()) is True

    def test_requests_timeout_is_retryable(self):
        assert _is_retryable(requests.exceptions.Timeout()) is True

    def test_aiohttp_server_timeout_is_retryable(self):
        assert _is_retryable(aiohttp.ServerTimeoutError("timeout")) is True

    def test_connection_error_is_retryable(self):
        assert _is_retryable(ConnectionError()) is True

    def test_requests_connection_error_is_retryable(self):
        assert _is_retryable(requests.exceptions.ConnectionError()) is True

    def test_aiohttp_client_connection_error_is_retryable(self):
        assert _is_retryable(aiohttp.ClientConnectionError()) is True

    def test_aiohttp_server_disconnected_is_retryable(self):
        assert _is_retryable(aiohttp.ServerDisconnectedError()) is True

    def test_requests_5xx_is_retryable(self):
        response = MagicMock()
        response.status_code = 503
        exc = requests.exceptions.HTTPError(response=response)
        assert _is_retryable(exc) is True

    def test_aiohttp_5xx_is_retryable(self):
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=500,
            message="Internal Server Error",
        )
        assert _is_retryable(exc) is True

    def test_requests_4xx_not_retryable(self):
        response = MagicMock()
        response.status_code = 404
        exc = requests.exceptions.HTTPError(response=response)
        assert _is_retryable(exc) is False

    def test_aiohttp_4xx_not_retryable(self):
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=404,
            message="Not Found",
        )
        assert _is_retryable(exc) is False

    def test_value_error_not_retryable(self):
        assert _is_retryable(ValueError("bad value")) is False

    def test_type_error_not_retryable(self):
        assert _is_retryable(TypeError("wrong type")) is False

    def test_requests_http_error_no_response_not_retryable(self):
        exc = requests.exceptions.HTTPError(response=None)
        assert _is_retryable(exc) is False


# =============================================================================
# HTTPConnection tests
# =============================================================================


class TestHTTPConnection:
    def test_init_defaults(self):
        conn = HTTPConnection()
        assert conn.reuse_client is True
        assert conn._sync_client is None
        assert conn._async_client is None

    def test_init_no_reuse(self):
        conn = HTTPConnection(reuse_client=False)
        assert conn.reuse_client is False

    def test_get_sync_client_creates_session(self):
        conn = HTTPConnection()
        client = conn.get_sync_client()
        assert isinstance(client, requests.Session)

    def test_get_sync_client_reuses_session(self):
        conn = HTTPConnection(reuse_client=True)
        client1 = conn.get_sync_client()
        client2 = conn.get_sync_client()
        assert client1 is client2

    def test_get_sync_client_no_reuse(self):
        conn = HTTPConnection(reuse_client=False)
        conn.get_sync_client()
        client2 = conn.get_sync_client()
        # When reuse_client is False, a new session is created each time
        assert client2 is not None

    def test_validate_http_url_valid(self):
        conn = HTTPConnection()
        # Should not raise
        conn._validate_http_url("http://example.com")
        conn._validate_http_url("https://example.com/path")

    def test_validate_http_url_invalid_scheme(self):
        conn = HTTPConnection()
        with pytest.raises(ValueError, match="Invalid HTTP URL"):
            conn._validate_http_url("ftp://example.com")

    def test_validate_http_url_no_scheme(self):
        conn = HTTPConnection()
        with pytest.raises(ValueError, match="Invalid HTTP URL"):
            conn._validate_http_url("example.com")

    def test_headers_contains_user_agent(self):
        conn = HTTPConnection()
        headers = conn._headers()
        assert "User-Agent" in headers
        assert "vLLM/" in headers["User-Agent"]

    def test_headers_with_extras(self):
        conn = HTTPConnection()
        headers = conn._headers(Authorization="Bearer token123")
        assert headers["Authorization"] == "Bearer token123"
        assert "User-Agent" in headers


class TestHTTPConnectionGetBytes:
    @patch("vllm.envs.VLLM_MEDIA_FETCH_MAX_RETRIES", 1)
    def test_get_bytes_invalid_url(self):
        conn = HTTPConnection()
        with pytest.raises(ValueError, match="Invalid HTTP URL"):
            conn.get_bytes("not-a-url")

    @patch("vllm.envs.VLLM_MEDIA_FETCH_MAX_RETRIES", 1)
    def test_get_bytes_connection_error(self):
        conn = HTTPConnection()
        with pytest.raises(requests.exceptions.ConnectionError):
            conn.get_bytes("http://192.0.2.1:1/nonexistent", timeout=0.1)


class TestHTTPConnectionAsync:
    @pytest.mark.asyncio
    async def test_get_async_client_creates_session(self):
        conn = HTTPConnection()
        client = await conn.get_async_client()
        assert isinstance(client, aiohttp.ClientSession)
        await client.close()

    @pytest.mark.asyncio
    async def test_get_async_client_reuses_session(self):
        conn = HTTPConnection()
        client1 = await conn.get_async_client()
        client2 = await conn.get_async_client()
        assert client1 is client2
        await client1.close()


# =============================================================================
# global_http_connection tests
# =============================================================================


class TestGlobalHTTPConnection:
    def test_global_instance_exists(self):
        assert global_http_connection is not None
        assert isinstance(global_http_connection, HTTPConnection)
