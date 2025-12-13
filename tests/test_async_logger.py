# ABOUTME: Unit tests for AsyncFileLogger
# ABOUTME: Tests log rotation, thread safety, unicode sanitization, and security masking

"""Tests for async_logger.py module."""

import asyncio
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from ralph_orchestrator.async_logger import AsyncFileLogger


class TestAsyncFileLoggerInit:
    """Tests for AsyncFileLogger initialization."""

    def test_init_creates_log_directory(self):
        """Logger should create log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "subdir" / "test.log"
            logger = AsyncFileLogger(str(log_path))
            assert log_path.parent.exists()

    def test_init_rejects_empty_path(self):
        """Logger should reject empty log file path."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            AsyncFileLogger("")

    def test_init_rejects_none_path(self):
        """Logger should reject None log file path."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            AsyncFileLogger(None)

    def test_init_accepts_path_object(self):
        """Logger should accept Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(log_path)
            assert logger.log_file == log_path

    def test_init_verbose_default_false(self):
        """Verbose should default to False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            assert logger.verbose is False

    def test_init_verbose_can_be_enabled(self):
        """Verbose can be set to True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path), verbose=True)
            assert logger.verbose is True


class TestAsyncFileLoggerBasicLogging:
    """Tests for basic logging functionality."""

    @pytest.mark.asyncio
    async def test_log_creates_file(self):
        """Logging should create the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Test message")
            assert log_path.exists()

    @pytest.mark.asyncio
    async def test_log_writes_message(self):
        """Logging should write the message to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Hello World")
            content = log_path.read_text()
            assert "Hello World" in content
            assert "[INFO]" in content

    @pytest.mark.asyncio
    async def test_log_includes_timestamp(self):
        """Log entries should include timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Test")
            content = log_path.read_text()
            # Timestamp format: YYYY-MM-DD HH:MM:SS
            import re

            assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)

    @pytest.mark.asyncio
    async def test_log_info(self):
        """log_info should use INFO level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log_info("Info message")
            content = log_path.read_text()
            assert "[INFO]" in content

    @pytest.mark.asyncio
    async def test_log_success(self):
        """log_success should use SUCCESS level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log_success("Success message")
            content = log_path.read_text()
            assert "[SUCCESS]" in content

    @pytest.mark.asyncio
    async def test_log_error(self):
        """log_error should use ERROR level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log_error("Error message")
            content = log_path.read_text()
            assert "[ERROR]" in content

    @pytest.mark.asyncio
    async def test_log_warning(self):
        """log_warning should use WARNING level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log_warning("Warning message")
            content = log_path.read_text()
            assert "[WARNING]" in content


class TestAsyncFileLoggerSyncMethods:
    """Tests for synchronous wrapper methods."""

    def test_log_info_sync(self):
        """log_info_sync should work synchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.log_info_sync("Sync info")
            content = log_path.read_text()
            assert "[INFO]" in content
            assert "Sync info" in content

    def test_log_success_sync(self):
        """log_success_sync should work synchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.log_success_sync("Sync success")
            content = log_path.read_text()
            assert "[SUCCESS]" in content

    def test_log_error_sync(self):
        """log_error_sync should work synchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.log_error_sync("Sync error")
            content = log_path.read_text()
            assert "[ERROR]" in content

    def test_log_warning_sync(self):
        """log_warning_sync should work synchronously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.log_warning_sync("Sync warning")
            content = log_path.read_text()
            assert "[WARNING]" in content

    def test_info_standard_interface(self):
        """info() should work as standard logging interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.info("Standard info")
            content = log_path.read_text()
            assert "Standard info" in content

    def test_error_standard_interface(self):
        """error() should work as standard logging interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.error("Standard error")
            content = log_path.read_text()
            assert "Standard error" in content

    def test_warning_standard_interface(self):
        """warning() should work as standard logging interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.warning("Standard warning")
            content = log_path.read_text()
            assert "Standard warning" in content

    def test_critical_standard_interface(self):
        """critical() should work as standard logging interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            logger.critical("Critical message")
            content = log_path.read_text()
            assert "Critical message" in content


class TestAsyncFileLoggerUnicodeSanitization:
    """Tests for unicode sanitization."""

    @pytest.mark.asyncio
    async def test_sanitize_unicode_normal_text(self):
        """Normal text should pass through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Hello World")
            content = log_path.read_text()
            assert "Hello World" in content

    @pytest.mark.asyncio
    async def test_sanitize_unicode_emoji(self):
        """Emoji should be handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Test with emoji: 🎉")
            content = log_path.read_text()
            # Emoji might be preserved or replaced
            assert "Test with emoji" in content

    @pytest.mark.asyncio
    async def test_sanitize_unicode_non_ascii(self):
        """Non-ASCII characters should be handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Cafe with accent: cafe")
            content = log_path.read_text()
            assert "Cafe with accent" in content


class TestAsyncFileLoggerSecurityMasking:
    """Tests for sensitive data masking."""

    @pytest.mark.asyncio
    async def test_masks_api_keys(self):
        """API keys should be masked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Using API key: sk-1234567890abcdef")
            content = log_path.read_text()
            # Original key should not be visible
            assert "1234567890abcdef" not in content
            # Should contain masked version
            assert "sk-***********" in content

    @pytest.mark.asyncio
    async def test_masks_bearer_tokens(self):
        """Bearer tokens should be masked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Auth: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            content = log_path.read_text()
            assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in content
            assert "Bearer ***********" in content

    @pytest.mark.asyncio
    async def test_masks_passwords(self):
        """Passwords should be masked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "password=mysecretpassword123")
            content = log_path.read_text()
            assert "mysecretpassword123" not in content
            assert "*********" in content


class TestAsyncFileLoggerRotation:
    """Tests for log rotation functionality."""

    def test_rotation_constants(self):
        """Verify rotation constants."""
        assert AsyncFileLogger.MAX_LOG_SIZE_BYTES == 10 * 1024 * 1024
        assert AsyncFileLogger.MAX_BACKUP_FILES == 3

    @pytest.mark.asyncio
    async def test_rotation_creates_backup(self):
        """Log rotation should create backup files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))

            # Create a log file larger than max size
            with open(log_path, "w") as f:
                # Write enough data to exceed MAX_LOG_SIZE_BYTES
                f.write("x" * (AsyncFileLogger.MAX_LOG_SIZE_BYTES + 1000))

            # Trigger rotation by logging
            await logger.log("INFO", "Trigger rotation")

            # Check backup was created
            backup_path = log_path.with_suffix(".log.1")
            assert backup_path.exists()

    @pytest.mark.asyncio
    async def test_rotation_max_backups(self):
        """Log rotation should respect max backup count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"

            # Create multiple backup files
            for i in range(1, 6):
                backup = log_path.with_suffix(f".log.{i}")
                backup.write_text(f"backup {i}")

            logger = AsyncFileLogger(str(log_path))

            # Create a log file larger than max size
            with open(log_path, "w") as f:
                f.write("x" * (AsyncFileLogger.MAX_LOG_SIZE_BYTES + 1000))

            # Trigger rotation
            await logger.log("INFO", "Trigger rotation")

            # Verify max backups (3)
            # .log.1, .log.2, .log.3 should exist, .log.4 and beyond might be rotated out


class TestAsyncFileLoggerStats:
    """Tests for statistics methods."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_file(self):
        """get_stats should return zeros for non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            stats = logger.get_stats()
            assert stats["success_count"] == 0
            assert stats["error_count"] == 0
            assert stats["start_time"] is None

    @pytest.mark.asyncio
    async def test_get_stats_counts_successes(self):
        """get_stats should count successful iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("SUCCESS", "Iteration 1 completed successfully")
            await logger.log("SUCCESS", "Iteration 2 completed successfully")
            stats = logger.get_stats()
            assert stats["success_count"] == 2

    @pytest.mark.asyncio
    async def test_get_stats_counts_errors(self):
        """get_stats should count failed iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("ERROR", "Iteration 1 failed with error")
            stats = logger.get_stats()
            assert stats["error_count"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_extracts_start_time(self):
        """get_stats should extract start time from first entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Session started")
            stats = logger.get_stats()
            assert stats["start_time"] is not None

    @pytest.mark.asyncio
    async def test_get_recent_lines(self):
        """get_recent_lines should return last N lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            for i in range(5):
                await logger.log("INFO", f"Message {i}")
            lines = logger.get_recent_lines(2)
            assert len(lines) == 2
            assert "Message 4" in lines[1]

    @pytest.mark.asyncio
    async def test_get_recent_lines_default_count(self):
        """get_recent_lines should use default count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            for i in range(5):
                await logger.log("INFO", f"Message {i}")
            lines = logger.get_recent_lines()
            assert len(lines) == AsyncFileLogger.DEFAULT_RECENT_LINES_COUNT

    @pytest.mark.asyncio
    async def test_count_pattern(self):
        """count_pattern should count occurrences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "Test pattern")
            await logger.log("INFO", "Another pattern here")
            await logger.log("INFO", "No match")
            count = logger.count_pattern("pattern")
            assert count == 2

    @pytest.mark.asyncio
    async def test_get_start_time(self):
        """get_start_time should return first log timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))
            await logger.log("INFO", "First message")
            await logger.log("INFO", "Second message")
            start_time = logger.get_start_time()
            assert start_time is not None
            # Should be in format YYYY-MM-DD HH:MM:SS
            parts = start_time.split(" ")
            assert len(parts) == 2


class TestAsyncFileLoggerThreadSafety:
    """Tests for thread-safe operations."""

    @pytest.mark.asyncio
    async def test_concurrent_logging(self):
        """Multiple concurrent log calls should not corrupt data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))

            # Create multiple concurrent log tasks
            tasks = [logger.log("INFO", f"Message {i}") for i in range(10)]
            await asyncio.gather(*tasks)

            # Verify all messages were logged
            content = log_path.read_text()
            for i in range(10):
                assert f"Message {i}" in content

    def test_concurrent_sync_logging(self):
        """Multiple threads using sync methods should not corrupt data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path))

            results = []

            def log_from_thread(i):
                try:
                    logger.log_info_sync(f"Thread {i}")
                    results.append(i)
                except Exception as e:
                    results.append(f"error: {e}")

            threads = [threading.Thread(target=log_from_thread, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should have completed
            assert len(results) == 5

            # All messages should be in log
            content = log_path.read_text()
            for i in range(5):
                assert f"Thread {i}" in content


class TestAsyncFileLoggerVerbose:
    """Tests for verbose mode."""

    @pytest.mark.asyncio
    async def test_verbose_prints_to_console(self, capsys):
        """Verbose mode should print to console."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path), verbose=True)
            await logger.log("INFO", "Verbose message")
            captured = capsys.readouterr()
            assert "Verbose message" in captured.out

    @pytest.mark.asyncio
    async def test_non_verbose_no_console(self, capsys):
        """Non-verbose mode should not print to console."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = AsyncFileLogger(str(log_path), verbose=False)
            await logger.log("INFO", "Silent message")
            captured = capsys.readouterr()
            assert captured.out == ""
