"""
Logging configuration for the self-planned pipeline.

This module provides centralized logging configuration that can be controlled
via a debug flag. In normal mode, only essential information is shown. In debug
mode, all verbose details are displayed.
"""

from typing import Any, Dict
import json


class PipelineLogger:
    """Central logger for the pipeline with debug mode support."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def set_debug(self, debug: bool):
        """Enable or disable debug mode."""
        self.debug = debug

    def info(self, message: str):
        """Always print important info messages."""
        print(message)

    def debug_print(self, message: str):
        """Print only if debug mode is enabled."""
        if self.debug:
            print(message)

    def section(self, title: str, width: int = 60):
        """Print a section header (always shown)."""
        print(f"\n{title}")
        print("=" * width)

    def subsection(self, title: str, width: int = 30):
        """Print a subsection header (always shown)."""
        print(f"\n{title}")
        print("-" * width)

    def success(self, message: str):
        """Print success message (always shown)."""
        print(f"✅ {message}")

    def warning(self, message: str):
        """Print warning message (always shown)."""
        print(f"⚠️  {message}")

    def error(self, message: str):
        """Print error message (always shown)."""
        print(f"❌ {message}")

    def stage_header(self, stage_id: str, reads: list, writes: list):
        """Print stage execution header (always shown)."""
        print(f"\n🔄 {stage_id}: {reads} → {writes}", end=" ")

    def stage_complete(self, execution_time: float, output_info: str):
        """Print stage completion info (always shown)."""
        print(f" ({execution_time:.1f}s, {output_info})")

    def stage_output_preview(self, preview: str):
        """Print stage output preview (always shown)."""
        print(f"     → {preview}")

    def model_call_separator(self, stage_id: str, show: bool = None):
        """Print model call separator (only in debug mode)."""
        if show is None:
            show = self.debug
        if show:
            print(f"\n{'='*80}")
            print(f"🤖 MODEL CALL - Stage: {stage_id}")
            print(f"{'='*80}")

    def raw_input(self, content: str, show: bool = None):
        """Print raw input to model (only in debug mode)."""
        if show is None:
            show = self.debug
        if show:
            print(f"📥 RAW INPUT:")
            print(f"{'-'*40}")
            print(content)
            print(f"{'-'*40}")

    def raw_output(self, content: str, show: bool = None):
        """Print raw output from model (only in debug mode)."""
        if show is None:
            show = self.debug
        if show:
            print(f"📤 RAW OUTPUT:")
            print(f"{'-'*40}")
            print(content)
            print(f"{'-'*40}")
            print(f"{'='*80}")

    def stage_details(self, message: str):
        """Print detailed stage information (only in debug mode)."""
        if self.debug:
            print(f"     {message}")

    def full_output(self, outputs: Dict[str, Any], always_show: bool = False):
        """Print full stage outputs (only in debug mode unless always_show=True)."""
        if self.debug or always_show:
            print(f"     📄 Full Output:")
            for key, value in outputs.items():
                print(f"       {key}: {value}")

    def stage_summary(self, prompt_preview: str, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        """Print compact stage summary (always shown in normal mode)."""
        print(f"\n     📝 Prompt: {prompt_preview}")
        print(f"     📥 Input: {self._format_compact(input_data)}")
        print(f"     📤 Output: {self._format_compact(output_data)}")

    def _format_compact(self, data: Dict[str, Any], max_len: int = 100) -> str:
        """Format data compactly for display."""
        if not data:
            return "{}"

        parts = []
        for key, value in data.items():
            if isinstance(value, str):
                preview = value[:50] + "..." if len(value) > 50 else value
                parts.append(f"{key}='{preview}'")
            elif isinstance(value, dict):
                parts.append(f"{key}={{{len(value)} keys}}")
            elif isinstance(value, list):
                parts.append(f"{key}=[{len(value)} items]")
            elif isinstance(value, bool):
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}={type(value).__name__}")

        result = ", ".join(parts)
        if len(result) > max_len:
            result = result[:max_len] + "..."
        return "{" + result + "}"

    def plan_structure(self, plan_json: str):
        """Print plan structure (only in debug mode)."""
        if self.debug:
            print("📋 Plan structure:")
            print(plan_json)

    def planning_progress(self, message: str, show_always: bool = False):
        """Print planning progress (shown unless very verbose)."""
        if show_always or self.debug:
            print(message)

    def summary_section(self):
        """Print summary section separator (always shown)."""
        print("\n" + "=" * 60)
        print("📋 PIPELINE SUMMARY")
        print("=" * 60)


# Global logger instance
_logger = None


def get_logger() -> PipelineLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = PipelineLogger(debug=False)
    return _logger


def init_logger(debug: bool = False) -> PipelineLogger:
    """Initialize the global logger with debug setting."""
    global _logger
    _logger = PipelineLogger(debug=debug)
    return _logger
