# src/mcp_cli/main.py
"""Entry-point for the MCP CLI."""

from __future__ import annotations
import asyncio
import atexit
import gc
import logging
import signal
import sys
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
# 1) Bring in the CLI‐side register_all_commands, which populates CommandRegistry
from mcp_cli.cli.commands import register_all_commands
from mcp_cli.cli.registry import CommandRegistry
from mcp_cli.run_command import run_command_sync
from mcp_cli.ui.ui_helpers import restore_terminal
from mcp_cli.cli_options import process_options
from mcp_cli.provider_config import ProviderConfig  # Import ProviderConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# Ensure terminal restoration on exit
atexit.register(restore_terminal)

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------
app = typer.Typer()


# ---------------------------------------------------------------------------
# Interactive Mode (special top‐level command)
# ---------------------------------------------------------------------------
@app.command("interactive")
def _interactive_command(
    config_file: str = "server_config.json",
    server: Optional[str] = None,
    provider: Optional[str] = None,  # Make provider optional
    model: Optional[str] = None,     # Make model optional
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL for the provider"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the provider"),
    disable_filesystem: bool = False,
):
    """Start interactive command mode."""
    # Load provider config to get saved settings
    provider_config = ProviderConfig()
    
    # Use command-line parameters if provided, otherwise use saved settings
    actual_provider = provider or provider_config.get_active_provider()
    actual_model = model or provider_config.get_active_model()

    servers, _, server_names = process_options(
        server, disable_filesystem, actual_provider, actual_model, config_file
    )

    # Import the new interactive shell entrypoint
    from mcp_cli.interactive.shell import interactive_mode

    run_command_sync(
        interactive_mode,
        config_file,
        servers,
        extra_params={
            "provider": actual_provider,
            "model": actual_model,
            "server_names": server_names,
            "api_base": api_base,
            "api_key": api_key,
        },
    )


# ---------------------------------------------------------------------------
# Register all CLI commands into the registry
# ---------------------------------------------------------------------------
register_all_commands()


# ---------------------------------------------------------------------------
# Wire up sub‐command groups
# ---------------------------------------------------------------------------
CommandRegistry.create_subcommand_group(
    app, "tools",     ["list", "call"], run_command_sync
)
CommandRegistry.create_subcommand_group(
    app, "resources", ["list"],         run_command_sync
)
CommandRegistry.create_subcommand_group(
    app, "prompts",   ["list"],         run_command_sync
)


# ---------------------------------------------------------------------------
# Register standalone top‐level commands
# ---------------------------------------------------------------------------
for name in ("ping", "chat", "cmd", "provider"):  # Added "provider" to standalone commands
    cmd = CommandRegistry.get_command(name)
    if cmd:
        cmd.register(app, run_command_sync)


# ---------------------------------------------------------------------------
# Make "chat" the default if no subcommand is supplied
# ---------------------------------------------------------------------------
chat_cmd = CommandRegistry.get_command("chat")
if chat_cmd:
    chat_cmd.register_as_default(app, run_command_sync)


# ---------------------------------------------------------------------------
# Signal‐handler for clean shutdown
# ---------------------------------------------------------------------------
def _signal_handler(sig, _frame):
    logging.debug("Received signal %s, restoring terminal", sig)
    restore_terminal()
    sys.exit(0)


def _setup_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _signal_handler)


# ---------------------------------------------------------------------------
# Main entry‐point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if sys.platform == "win32":
        # Use the selector event loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    _setup_signal_handlers()
    try:
        app()
    finally:
        restore_terminal()
        gc.collect()