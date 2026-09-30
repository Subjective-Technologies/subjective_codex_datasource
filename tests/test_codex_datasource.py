"""
Test script for SubjectiveCodexDataSource.

Usage:
    python test_codex_datasource.py

Requirements:
    - OpenAI Codex CLI installed (https://developers.openai.com/codex/cli/)
    - Either OPENAI_API_KEY environment variable or OAuth authentication
"""

import base64
import os
import re
import sys
import unittest
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
_SERVICE = os.path.dirname(os.path.dirname(_PLUGIN))
for _path in (
    _PLUGIN,
    os.path.join(_SERVICE, "libs", "dependencies", "subjective-abstract-data-source-package"),
    os.path.join(_SERVICE, "libs", "dependencies", "brainboost_data_source_logger_package"),
    os.path.join(_SERVICE, "libs", "dependencies", "brainboost_configuration_package"),
):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from SubjectiveCodexDataSource import SubjectiveCodexDataSource


def _skip_live_when_pytest_collects():
    """This file is also a manual script. Pytest must not start the Codex CLI."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest

        pytest.skip("manual live script; the unit suite must not start Codex")


def test_installation_check():
    """Test checking Codex CLI installation."""
    _skip_live_when_pytest_collects()
    print("=" * 50)
    print("Testing Codex CLI Installation Check")
    print("=" * 50)

    datasource = SubjectiveCodexDataSource(
        name="test_codex",
        params={"async_mode": False}
    )

    status = datasource.check_codex_installation()
    print(f"Installation status: {status}")

    if not status.get("installed"):
        print("\nCodex CLI is not installed!")
        print("Install from: https://developers.openai.com/codex/cli/")
        return False

    print(f"Codex CLI found at: {status.get('path')}")
    print(f"Version: {status.get('version')}")
    return True


def test_connection_data():
    """Test get_connection_data method."""
    print("\n" + "=" * 50)
    print("Testing Connection Data")
    print("=" * 50)

    datasource = SubjectiveCodexDataSource(
        name="test_codex",
        params={}
    )

    connection_data = datasource.get_connection_data()
    print(f"Connection type: {connection_data['connection_type']}")
    print(f"Number of fields: {len(connection_data['fields'])}")

    print("\nAvailable fields:")
    for field in connection_data['fields']:
        required = "required" if field.get("required") else "optional"
        print(f"  - {field['name']} ({field['type']}, {required})")


def test_icon():
    """Test get_icon method."""
    print("\n" + "=" * 50)
    print("Testing Icon")
    print("=" * 50)

    datasource = SubjectiveCodexDataSource(
        name="test_codex",
        params={}
    )

    icon = datasource.get_icon()
    if icon and icon.strip().startswith("<svg"):
        print("Icon SVG loaded successfully")
        print(f"Icon length: {len(icon)} characters")
    else:
        print("Warning: Icon not loaded properly")


def test_api_key_auth():
    """Test with API key authentication."""
    _skip_live_when_pytest_collects()
    print("\n" + "=" * 50)
    print("Testing API Key Authentication")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping API key test")
        return

    datasource = SubjectiveCodexDataSource(
        name="test_codex_api",
        params={
            "async_mode": False,
            "auth_method": "api_key",
            "api_key": api_key,
            "model": "o4-mini",
            "sandbox_mode": "read-only",
            "timeout": 60
        }
    )

    print("Sending test message...")
    response = datasource.send_message("Say 'Hello from Codex!' and nothing else.")

    if response:
        if response.get("success"):
            print(f"Response: {response.get('response')}")
        else:
            print(f"Error: {response.get('message')}")
    else:
        print("No response received")


def test_sync_mode():
    """Test synchronous message processing."""
    _skip_live_when_pytest_collects()
    print("\n" + "=" * 50)
    print("Testing Sync Mode")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping sync mode test")
        return

    datasource = SubjectiveCodexDataSource(
        name="test_codex_sync",
        params={
            "async_mode": False,
            "auth_method": "api_key",
            "api_key": api_key,
            "model": "o4-mini",
            "sandbox_mode": "read-only"
        }
    )

    # Test simple prompt
    print("Sending: 'What is 2 + 2?'")
    response = datasource.send_message("What is 2 + 2? Reply with just the number.")

    if response and response.get("success"):
        print(f"Response: {response.get('response')}")

        # Check conversation history
        history = datasource.get_conversation_history()
        print(f"Conversation history entries: {len(history)}")
    else:
        print(f"Error: {response}")


def test_async_mode():
    """Test asynchronous message processing."""
    _skip_live_when_pytest_collects()
    print("\n" + "=" * 50)
    print("Testing Async Mode")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping async mode test")
        return

    responses = []

    def on_response(response):
        responses.append(response)
        if response.get("success"):
            print(f"Async response received: {response.get('response')[:50]}...")
        else:
            print(f"Async error: {response.get('message')}")

    datasource = SubjectiveCodexDataSource(
        name="test_codex_async",
        params={
            "async_mode": True,
            "auth_method": "api_key",
            "api_key": api_key,
            "model": "o4-mini",
            "sandbox_mode": "read-only"
        }
    )

    datasource.set_response_callback(on_response)

    print("Sending async message...")
    datasource.send_message("Count from 1 to 5, one number per line.")

    print("Waiting for response...")
    datasource.wait_for_pending(timeout=120)

    print(f"Total responses received: {len(responses)}")

    datasource.stop()


def _paths_named(text: str) -> list[str]:
    """Absolute paths the prompt actually contains. No tag name is assumed."""
    return re.findall(r"(/[^\s\"'<>]+)", text)


class DashboardFileDeliveryTests(unittest.TestCase):
    """Files in the dashboard shape must survive into the Codex command.

    `codex exec --image` carries image files. Text is inlined in the prompt, which
    is the last argument. A non-image file is written and named by an absolute path
    in that prompt. Bytes are read while `subprocess.run` is in progress, because
    the temp dir is removed when the call returns.
    """

    def _dashboard(self, filename: str, mime_type: str, raw: bytes) -> dict:
        return {
            "filename": filename,
            "mime_type": mime_type,
            "content": base64.b64encode(raw).decode("ascii"),
        }

    def _drive(self, content: str, files: list[dict]) -> list:
        previous = os.environ.get("OPENAI_API_KEY")
        source = SubjectiveCodexDataSource(
            name="file-delivery",
            params={
                "async_mode": False,
                "auth_method": "api_key",
                "api_key": "test-not-a-real-key",
                "working_directory": "/tmp",
            },
        )
        source._codex_path = "/usr/bin/codex"
        captured: dict = {}

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = list(cmd)
            prompt = str(cmd[-1])
            staged = {}
            for item in [*cmd, *_paths_named(prompt)]:
                if isinstance(item, str) and os.path.isfile(item):
                    with open(item, "rb") as handle:
                        staged[item] = handle.read()
            captured["staged"] = staged
            return subprocess_result()

        try:
            with patch("SubjectiveCodexDataSource.subprocess.run", side_effect=fake_run):
                result = source.handle_message(content, files=files)
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous
        self.assertIn("cmd", captured, result)
        return captured["cmd"], captured.get("staged", {})

    def test_image_png_bytes_are_on_a_path_the_command_emits(self):
        raw = b"\x89PNG\r\n\x1a\nfake-image"
        cmd, staged = self._drive("Look", [self._dashboard("pixel.png", "image/png", raw)])
        self.assertIn("--image", cmd)
        self.assertTrue(
            any(data == raw for data in staged.values()),
            "image/png did not travel: the codex command names no path with those bytes. "
            f"command={cmd!r}",
        )

    def test_text_plain_is_inlined_in_the_prompt(self):
        body = "ship the notes"
        cmd, _staged = self._drive("Look", [self._dashboard("notes.txt", "text/plain", body.encode("utf-8"))])
        prompt = str(cmd[-1])
        self.assertIn(body, prompt, f"text/plain was not inlined in the prompt: {prompt!r}")
        self.assertIn("<attachment name=", prompt)
        self.assertNotEqual(prompt.strip(), "notes.txt")
        self.assertFalse(prompt.strip().endswith("/notes.txt"), prompt)

    def test_pdf_bytes_are_on_a_path_the_command_emits(self):
        raw = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        cmd, staged = self._drive("Look", [self._dashboard("spec.pdf", "application/pdf", raw)])
        prompt = str(cmd[-1])
        self.assertIn("<attachment_path path=", prompt)
        self.assertTrue(
            any(data == raw for data in staged.values()),
            "application/pdf did not travel: the codex command names no path with those bytes. "
            f"command={cmd!r}",
        )


def subprocess_result():
    import subprocess

    return subprocess.CompletedProcess(args=["codex"], returncode=0, stdout="", stderr="")


def main():
    """Run all tests."""
    print("SubjectiveCodexDataSource Test Suite")
    print("=" * 50)

    # Basic tests (no API key required)
    installed = test_installation_check()
    test_connection_data()
    test_icon()

    if not installed:
        print("\nSkipping functional tests - Codex CLI not installed")
        return

    # Functional tests (require API key)
    test_api_key_auth()
    test_sync_mode()
    test_async_mode()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
