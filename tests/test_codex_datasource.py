"""
Test script for SubjectiveCodexDataSource.

Usage:
    python test_codex_datasource.py

Requirements:
    - OpenAI Codex CLI installed (https://developers.openai.com/codex/cli/)
    - Either OPENAI_API_KEY environment variable or OAuth authentication
"""

import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SubjectiveCodexDataSource import SubjectiveCodexDataSource


def test_installation_check():
    """Test checking Codex CLI installation."""
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
