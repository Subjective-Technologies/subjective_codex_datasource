import os
import subprocess
import json
import shutil
from typing import Any, Optional
from subjective_abstract_data_source_package import SubjectiveOnDemandDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveCodexDataSource(SubjectiveOnDemandDataSource):
    """
    OnDemand data source for OpenAI Codex CLI interactions.

    Supports both API key and OAuth authentication methods.
    Uses 'codex exec' for stateless message processing.
    """

    def __init__(self, name=None, session=None, dependency_data_sources=None,
                 subscribers=None, params=None):
        super().__init__(
            name=name,
            session=session,
            dependency_data_sources=dependency_data_sources,
            subscribers=subscribers,
            params=params
        )

        # Authentication settings
        self.auth_method = self.params.get("auth_method", "api_key")
        self.api_key = self.params.get("api_key", "")

        # Codex CLI settings
        self.model = self.params.get("model", "o4-mini")
        self.sandbox_mode = self.params.get("sandbox_mode", "read-only")
        self.working_directory = self.params.get("working_directory", os.getcwd())
        self.timeout = self.params.get("timeout", 300)  # 5 minutes default

        # Additional options
        self.full_auto = self.params.get("full_auto", False)
        self.enable_search = self.params.get("enable_search", False)

        # Track authentication status
        self._authenticated = False
        self._codex_path = None

    def _find_codex_cli(self) -> Optional[str]:
        """Find the codex CLI executable."""
        if self._codex_path:
            return self._codex_path

        # Check if codex is in PATH
        codex_path = shutil.which("codex")
        if codex_path:
            self._codex_path = codex_path
            return codex_path

        # Check common installation locations
        common_paths = [
            os.path.expanduser("~/.local/bin/codex"),
            os.path.expanduser("~/bin/codex"),
            "/usr/local/bin/codex",
            "C:\\Program Files\\codex\\codex.exe",
            os.path.expanduser("~\\AppData\\Local\\Programs\\codex\\codex.exe"),
        ]

        for path in common_paths:
            if os.path.isfile(path):
                self._codex_path = path
                return path

        return None

    def _ensure_authenticated(self) -> bool:
        """Ensure the Codex CLI is authenticated."""
        if self._authenticated:
            return True

        codex_path = self._find_codex_cli()
        if not codex_path:
            BBLogger.log("Codex CLI not found. Please install it first.")
            return False

        if self.auth_method == "api_key" and self.api_key:
            # API key auth - set environment variable
            os.environ["OPENAI_API_KEY"] = self.api_key
            self._authenticated = True
            BBLogger.log("Using API key authentication for Codex CLI")
            return True

        elif self.auth_method == "oauth":
            # Trigger OAuth login; Codex will reuse an existing session if present.
            BBLogger.log("OAuth authentication required. Triggering Codex login...")
            return self._trigger_oauth_login()

        return False

    def _trigger_oauth_login(self) -> bool:
        """Trigger the OAuth login flow for Codex CLI."""
        codex_path = self._find_codex_cli()
        if not codex_path:
            return False

        try:
            # Use device-auth for better compatibility
            BBLogger.log("Starting Codex OAuth login (device auth)...")
            result = subprocess.run(
                [codex_path, "login", "--device-auth"],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes for login
            )

            if result.returncode == 0:
                self._authenticated = True
                output = (result.stdout or "").strip()
                if result.stderr:
                    output = "\n".join([output, result.stderr.strip()]).strip()
                if output:
                    BBLogger.log(f"Codex login output:\n{output}")
                BBLogger.log("Codex OAuth login successful")
                return True
            else:
                if result.stderr:
                    BBLogger.log(f"Codex login error:\n{result.stderr.strip()}")
                BBLogger.log("Codex OAuth login failed")
                return False

        except subprocess.TimeoutExpired:
            BBLogger.log("Codex OAuth login timed out")
            return False
        except Exception as e:
            BBLogger.log(f"Error during Codex OAuth login: {e}")
            return False

    def _build_command(self, message: str) -> list:
        """Build the codex exec command with all options."""
        codex_path = self._find_codex_cli()
        if not codex_path:
            raise RuntimeError("Codex CLI not found")

        cmd = [codex_path, "exec"]

        # Add JSON output for parsing
        cmd.append("--json")

        # Add model selection
        if self.model:
            cmd.extend(["--model", self.model])

        # Add sandbox mode
        if self.sandbox_mode:
            cmd.extend(["--sandbox", self.sandbox_mode])

        # Add full-auto mode if enabled
        if self.full_auto:
            cmd.append("--full-auto")

        # Add search if enabled
        if self.enable_search:
            cmd.append("--search")

        # Add working directory
        if self.working_directory:
            cmd.extend(["--cd", self.working_directory])

        # Add the prompt
        cmd.append(message)

        return cmd

    def _parse_json_output(self, output: str) -> dict:
        """Parse the newline-delimited JSON output from codex exec."""
        events = []
        assistant_message = ""

        for line in output.strip().split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)

                # Extract assistant message from events
                if event.get("type") == "message" and event.get("role") == "assistant":
                    content = event.get("content", [])
                    for item in content:
                        if item.get("type") == "text":
                            assistant_message += item.get("text", "")

            except json.JSONDecodeError:
                # Non-JSON line, might be status output
                continue

        return {
            "events": events,
            "assistant_message": assistant_message,
            "raw_output": output
        }

    def _process_message(self, message: Any) -> Any:
        """
        Process an incoming message using Codex CLI.

        Args:
            message: The prompt/message to send to Codex

        Returns:
            Dictionary with response data
        """
        # Ensure we have string message
        if isinstance(message, dict):
            message = message.get("content", str(message))
        message = str(message)

        # Ensure authentication
        if not self._ensure_authenticated():
            return {
                "error": True,
                "error_type": "authentication_error",
                "message": "Failed to authenticate with Codex CLI. Please check your credentials.",
                "original_message": message
            }

        try:
            # Build and execute command
            cmd = self._build_command(message)
            BBLogger.log(f"Executing Codex command: {' '.join(cmd[:3])}...")

            # Set up environment
            env = os.environ.copy()
            if self.auth_method == "api_key" and self.api_key:
                env["OPENAI_API_KEY"] = self.api_key

            # Execute codex
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=self.working_directory if os.path.isdir(self.working_directory) else None
            )

            if result.returncode != 0:
                BBLogger.log(f"Codex exec failed with code {result.returncode}: {result.stderr}")
                return {
                    "error": True,
                    "error_type": "execution_error",
                    "message": result.stderr or "Codex execution failed",
                    "return_code": result.returncode,
                    "original_message": message
                }

            # Parse the JSON output
            parsed = self._parse_json_output(result.stdout)

            return {
                "success": True,
                "response": parsed["assistant_message"],
                "events": parsed["events"],
                "original_message": message
            }

        except subprocess.TimeoutExpired:
            BBLogger.log(f"Codex command timed out after {self.timeout} seconds")
            return {
                "error": True,
                "error_type": "timeout",
                "message": f"Codex execution timed out after {self.timeout} seconds",
                "original_message": message
            }
        except Exception as e:
            BBLogger.log(f"Error executing Codex command: {e}")
            return {
                "error": True,
                "error_type": "exception",
                "message": str(e),
                "original_message": message
            }

    def check_codex_installation(self) -> dict:
        """
        Check if Codex CLI is installed and return status information.

        Returns:
            Dictionary with installation status and version info
        """
        codex_path = self._find_codex_cli()

        if not codex_path:
            return {
                "installed": False,
                "message": "Codex CLI not found. Please install from https://developers.openai.com/codex/cli/"
            }

        try:
            result = subprocess.run(
                [codex_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            return {
                "installed": True,
                "path": codex_path,
                "version": result.stdout.strip() if result.returncode == 0 else "unknown"
            }
        except Exception as e:
            return {
                "installed": True,
                "path": codex_path,
                "version": "unknown",
                "error": str(e)
            }

    def get_icon(self) -> str:
        """Return the SVG icon for Codex data source."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception as e:
            BBLogger.log(f"Error reading icon file: {e}")

        # Fallback SVG icon (OpenAI-style)
        return '''<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="24" height="24" rx="4" fill="#10A37F"/>
            <path d="M12 4C7.58 4 4 7.58 4 12C4 16.42 7.58 20 12 20C16.42 20 20 16.42 20 12C20 7.58 16.42 4 12 4ZM12 18C8.69 18 6 15.31 6 12C6 8.69 8.69 6 12 6C15.31 6 18 8.69 18 12C18 15.31 15.31 18 12 18Z" fill="white"/>
            <path d="M12 8V12L15 13.5" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <text x="12" y="16" text-anchor="middle" fill="white" font-size="4" font-family="Arial">CLI</text>
        </svg>'''

    def get_connection_data(self) -> dict:
        """Return connection configuration metadata."""
        return {
            "connection_type": "ON_DEMAND",
            "fields": [
                {
                    "name": "auth_method",
                    "type": "select",
                    "label": "Authentication Method",
                    "required": True,
                    "default": "api_key",
                    "description": "OAuth uses the local Codex CLI device login flow and reuses any existing session.",
                    "options": [
                        {"value": "api_key", "label": "API Key"},
                        {"value": "oauth", "label": "OAuth (Browser Login)"}
                    ]
                },
                {
                    "name": "api_key",
                    "type": "password",
                    "label": "OpenAI API Key (API Key auth only)",
                    "required": False,
                    "description": "Ignored when OAuth is selected.",
                    "depends_on": {"field": "auth_method", "value": "api_key"}
                },
                {
                    "name": "model",
                    "type": "select",
                    "label": "Model",
                    "required": False,
                    "default": "o4-mini",
                    "options": [
                        {"value": "o4-mini", "label": "O4 Mini (Fast)"},
                        {"value": "o3", "label": "O3 (Powerful)"},
                        {"value": "gpt-4.1", "label": "GPT-4.1"}
                    ]
                },
                {
                    "name": "sandbox_mode",
                    "type": "select",
                    "label": "Sandbox Mode",
                    "required": False,
                    "default": "read-only",
                    "options": [
                        {"value": "read-only", "label": "Read Only (Safest)"},
                        {"value": "workspace-write", "label": "Workspace Write"},
                        {"value": "danger-full-access", "label": "Full Access (Dangerous)"}
                    ]
                },
                {
                    "name": "working_directory",
                    "type": "text",
                    "label": "Working Directory",
                    "required": False,
                    "description": "Directory for Codex to operate in"
                },
                {
                    "name": "timeout",
                    "type": "number",
                    "label": "Timeout (seconds)",
                    "required": False,
                    "default": 300,
                    "description": "Maximum time to wait for Codex response"
                },
                {
                    "name": "full_auto",
                    "type": "checkbox",
                    "label": "Full Auto Mode",
                    "required": False,
                    "default": False,
                    "description": "Enable automatic approval for commands"
                },
                {
                    "name": "enable_search",
                    "type": "checkbox",
                    "label": "Enable Web Search",
                    "required": False,
                    "default": False,
                    "description": "Allow Codex to search the web"
                }
            ]
        }
