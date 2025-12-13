# ABOUTME: Unit tests for ACPAdapter class
# ABOUTME: Tests initialization, availability check, and session flow

"""Tests for ACPAdapter - the main ACP adapter for Ralph Orchestrator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest
import shutil

from ralph_orchestrator.adapters.acp import ACPAdapter


class TestACPAdapterInitialization:
    """Tests for ACPAdapter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        adapter = ACPAdapter()

        assert adapter.name == "acp"
        assert adapter.agent_command == "gemini"
        assert adapter.agent_args == []
        assert adapter.timeout == 300
        assert adapter.permission_mode == "auto_approve"
        assert adapter._client is None
        assert adapter._session_id is None
        assert adapter._initialized is False

    def test_init_with_custom_command(self):
        """Test initialization with custom agent command."""
        adapter = ACPAdapter(agent_command="custom-agent")

        assert adapter.agent_command == "custom-agent"

    def test_init_with_custom_args(self):
        """Test initialization with custom agent arguments."""
        adapter = ACPAdapter(agent_args=["--verbose", "--debug"])

        assert adapter.agent_args == ["--verbose", "--debug"]

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        adapter = ACPAdapter(timeout=600)

        assert adapter.timeout == 600

    def test_init_with_permission_mode(self):
        """Test initialization with custom permission mode."""
        adapter = ACPAdapter(permission_mode="deny_all")

        assert adapter.permission_mode == "deny_all"

    def test_init_from_config(self):
        """Test initialization from ACPAdapterConfig."""
        from ralph_orchestrator.adapters.acp_models import ACPAdapterConfig

        config = ACPAdapterConfig(
            agent_command="test-agent",
            agent_args=["--mode", "test"],
            timeout=120,
            permission_mode="allowlist",
            permission_allowlist=["fs/read_text_file"],
        )

        adapter = ACPAdapter.from_config(config)

        assert adapter.agent_command == "test-agent"
        assert adapter.agent_args == ["--mode", "test"]
        assert adapter.timeout == 120
        assert adapter.permission_mode == "allowlist"


class TestACPAdapterAvailability:
    """Tests for check_availability method."""

    def test_availability_when_command_exists(self):
        """Test availability returns True when command exists."""
        with patch.object(shutil, "which", return_value="/usr/bin/gemini"):
            adapter = ACPAdapter()
            assert adapter.check_availability() is True

    def test_availability_when_command_missing(self):
        """Test availability returns False when command missing."""
        with patch.object(shutil, "which", return_value=None):
            adapter = ACPAdapter()
            assert adapter.check_availability() is False

    def test_availability_checks_correct_command(self):
        """Test availability checks the configured command."""
        with patch.object(shutil, "which") as mock_which:
            mock_which.return_value = "/usr/bin/custom-agent"
            adapter = ACPAdapter(agent_command="custom-agent")
            adapter.check_availability()

            mock_which.assert_called_with("custom-agent")


class TestACPAdapterInitialize:
    """Tests for _initialize async method."""

    def _create_mock_client(self, init_response: dict, session_response: dict):
        """Helper to create a properly mocked ACPClient."""
        mock_client = MagicMock()
        mock_client.is_running = True
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.on_notification = MagicMock()
        mock_client.on_request = MagicMock()

        # Create futures for each request
        init_future = asyncio.Future()
        init_future.set_result(init_response)

        session_future = asyncio.Future()
        session_future.set_result(session_response)

        mock_client.send_request = MagicMock(side_effect=[init_future, session_future])

        return mock_client

    @pytest.mark.asyncio
    async def test_initialize_starts_client(self):
        """Test _initialize starts the ACP client."""
        adapter = ACPAdapter()

        mock_client = self._create_mock_client(
            {"protocolVersion": "2024-01", "capabilities": {}},
            {"sessionId": "test-session-123"},
        )

        with patch("ralph_orchestrator.adapters.acp.ACPClient", return_value=mock_client):
            await adapter._initialize()

            mock_client.start.assert_called_once()
            assert adapter._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_sends_initialize_request(self):
        """Test _initialize sends initialize request with protocol version."""
        adapter = ACPAdapter()

        mock_client = self._create_mock_client(
            {"protocolVersion": "2024-01", "capabilities": {}},
            {"sessionId": "test-session-123"},
        )

        with patch("ralph_orchestrator.adapters.acp.ACPClient", return_value=mock_client):
            await adapter._initialize()

            # Check initialize request was sent
            calls = mock_client.send_request.call_args_list
            assert len(calls) >= 1
            assert calls[0][0][0] == "initialize"
            assert "protocolVersion" in calls[0][0][1]

    @pytest.mark.asyncio
    async def test_initialize_creates_session(self):
        """Test _initialize creates new session and stores session_id."""
        adapter = ACPAdapter()

        mock_client = self._create_mock_client(
            {"protocolVersion": "2024-01", "capabilities": {}},
            {"sessionId": "test-session-abc"},
        )

        with patch("ralph_orchestrator.adapters.acp.ACPClient", return_value=mock_client):
            await adapter._initialize()

            # Check session/new was called
            calls = mock_client.send_request.call_args_list
            assert len(calls) >= 2
            assert calls[1][0][0] == "session/new"

            # Check session ID was stored
            assert adapter._session_id == "test-session-abc"

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test _initialize is idempotent (safe to call multiple times)."""
        adapter = ACPAdapter()
        adapter._initialized = True
        adapter._session_id = "existing-session"

        # Should not reinitialize
        await adapter._initialize()

        # Client should not be created
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_initialize_registers_notification_handler(self):
        """Test _initialize registers notification handler for updates."""
        adapter = ACPAdapter()

        mock_client = self._create_mock_client(
            {"protocolVersion": "2024-01"},
            {"sessionId": "test-session"},
        )

        with patch("ralph_orchestrator.adapters.acp.ACPClient", return_value=mock_client):
            await adapter._initialize()

            # Check notification handler was registered
            mock_client.on_notification.assert_called()


class TestACPAdapterExecute:
    """Tests for execute and aexecute methods."""

    def test_execute_when_unavailable(self):
        """Test execute returns error when adapter unavailable."""
        adapter = ACPAdapter()
        adapter.available = False

        response = adapter.execute("test prompt")

        assert response.success is False
        assert "not available" in response.error.lower()

    @pytest.mark.asyncio
    async def test_aexecute_when_unavailable(self):
        """Test aexecute returns error when adapter unavailable."""
        adapter = ACPAdapter()
        adapter.available = False

        response = await adapter.aexecute("test prompt")

        assert response.success is False
        assert "not available" in response.error.lower()

    @pytest.mark.asyncio
    async def test_aexecute_initializes_if_needed(self):
        """Test aexecute calls _initialize if not initialized."""
        adapter = ACPAdapter()
        adapter.available = True

        with patch.object(adapter, "_initialize", new_callable=AsyncMock) as mock_init:
            with patch.object(adapter, "_execute_prompt", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = MagicMock(
                    success=True, output="test", error=None
                )

                await adapter.aexecute("test prompt")

                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexecute_enhances_prompt(self):
        """Test aexecute enhances prompt with orchestration instructions."""
        adapter = ACPAdapter()
        adapter.available = True
        adapter._initialized = True
        adapter._session_id = "test-session"

        captured_prompt = None

        async def capture_prompt(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            from ralph_orchestrator.adapters.base import ToolResponse
            return ToolResponse(success=True, output="done")

        with patch.object(adapter, "_execute_prompt", side_effect=capture_prompt):
            await adapter.aexecute("simple task")

            # Should contain orchestration context
            assert captured_prompt is not None
            assert "ORCHESTRATION CONTEXT:" in captured_prompt

    def test_execute_runs_aexecute_sync(self):
        """Test sync execute wraps async aexecute."""
        adapter = ACPAdapter()
        adapter.available = True
        adapter._initialized = True
        adapter._session_id = "test-session"

        with patch.object(adapter, "_execute_prompt", new_callable=AsyncMock) as mock_exec:
            from ralph_orchestrator.adapters.base import ToolResponse
            mock_exec.return_value = ToolResponse(success=True, output="sync result")

            response = adapter.execute("test prompt")

            assert response.success is True
            assert response.output == "sync result"


class TestACPAdapterSignalHandling:
    """Tests for signal handling and shutdown."""

    def test_signal_handler_registration(self):
        """Test signal handlers are registered on init."""
        with patch("signal.signal") as mock_signal:
            adapter = ACPAdapter()

            # Should register SIGINT and SIGTERM handlers
            assert mock_signal.called

    @pytest.mark.asyncio
    async def test_shutdown_stops_client(self):
        """Test shutdown stops the ACP client."""
        adapter = ACPAdapter()

        mock_client = AsyncMock()
        adapter._client = mock_client
        adapter._initialized = True

        await adapter._shutdown()

        mock_client.stop.assert_called_once()
        assert adapter._initialized is False

    def test_kill_subprocess_sync(self):
        """Test sync subprocess kill for signal handlers."""
        adapter = ACPAdapter()

        mock_client = MagicMock()
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_client._process = mock_process
        adapter._client = mock_client

        adapter.kill_subprocess_sync()

        mock_process.terminate.assert_called_once()


class TestACPAdapterMetadata:
    """Tests for adapter metadata and string representation."""

    def test_str_representation(self):
        """Test string representation of adapter."""
        adapter = ACPAdapter(agent_command="test-agent")
        adapter.available = True

        result = str(adapter)

        assert "acp" in result
        assert "available: True" in result

    def test_estimate_cost(self):
        """Test cost estimation returns 0 (no billing info from ACP)."""
        adapter = ACPAdapter()

        cost = adapter.estimate_cost("test prompt")

        assert cost == 0.0
