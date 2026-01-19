"""Atomic state persistence with rolling backups."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from ai_orchestrator.core.workflow_phases import WorkflowState

logger = logging.getLogger(__name__)


class StateManager:
    """
    Crash-safe state persistence with rolling backups.

    Uses atomic writes (temp file + fsync + rename) to prevent corruption.
    Maintains rolling backups for recovery.
    """

    STATE_DIR = ".ai_orchestrator"
    STATE_FILE = "state.json"
    BACKUP_DIR = "backups"
    MAX_BACKUPS = 10

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.state_dir = self.project_root / self.STATE_DIR
        self.state_file = self.state_dir / self.STATE_FILE
        self.backup_dir = self.state_dir / self.BACKUP_DIR

        # Ensure directories exist
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def save_state_atomic(self, state: WorkflowState) -> None:
        """
        Save state atomically with backup.

        Uses temp file + fsync + rename pattern for crash safety.

        Args:
            state: The workflow state to save.
        """
        # Update timestamp
        state.updated_at = datetime.now(UTC)

        # Serialize to JSON
        state_dict = state.to_checkpoint_dict()
        state_json = json.dumps(state_dict, indent=2, default=str)

        # Create backup of current state if it exists
        if self.state_file.exists():
            self._create_backup()

        # Atomic write: temp file + fsync + rename
        temp_file = self.state_file.with_suffix(".tmp")

        try:
            # Write to temp file
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(state_json)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (works on Windows too with Path.replace)
            temp_file.replace(self.state_file)

            # Sync directory on POSIX to ensure directory entry is persisted
            # This guarantees the file name pointing to the new inode is durable
            if os.name == "posix":
                try:
                    dir_fd = os.open(str(self.state_dir), os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError as dir_sync_error:
                    # Log but don't fail - file itself is saved
                    logger.debug("Directory sync skipped: %s", dir_sync_error)

            logger.debug("State saved to %s", self.state_file)

        except Exception as e:
            # Clean up temp file on failure
            if temp_file.exists():
                temp_file.unlink()
            logger.error("Failed to save state: %s", e, exc_info=True)
            raise

    async def load_state(self) -> WorkflowState | None:
        """
        Load state from file, falling back to backups if needed.

        Returns:
            WorkflowState if found, None otherwise.
        """
        # Try main state file first
        if self.state_file.exists():
            try:
                state = self._load_from_file(self.state_file)
                logger.info("Loaded state from %s", self.state_file)
                return state
            except Exception as e:
                logger.warning("Failed to load main state file: %s", e)

        # Try backups in reverse chronological order
        backups = sorted(self.backup_dir.glob("state_*.json"), reverse=True)
        for backup in backups:
            try:
                state = self._load_from_file(backup)
                logger.info("Loaded state from backup %s", backup.name)
                return state
            except Exception as e:
                logger.warning("Failed to load backup %s: %s", backup.name, e)

        logger.info("No existing state found")
        return None

    async def save_checkpoint(
        self,
        state: WorkflowState,
        reason: str | None = None,
    ) -> str:
        """
        Save a named checkpoint for manual recovery.

        Args:
            state: The workflow state to checkpoint.
            reason: Optional reason for the checkpoint.

        Returns:
            Checkpoint ID.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"cp_{timestamp}"

        # Set checkpoint ID in state
        state.checkpoint_id = checkpoint_id

        # Create checkpoint directory
        checkpoints_dir = self.state_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        # Save checkpoint
        checkpoint_file = checkpoints_dir / f"{checkpoint_id}.json"
        state_dict = state.to_checkpoint_dict()

        if reason:
            state_dict["checkpoint_reason"] = reason

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)

        logger.info("Created checkpoint %s (reason: %s)", checkpoint_id, reason)
        return checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowState | None:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to load.

        Returns:
            WorkflowState if found, None otherwise.
        """
        checkpoints_dir = self.state_dir / "checkpoints"
        checkpoint_file = checkpoints_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            logger.warning("Checkpoint %s not found", checkpoint_id)
            return None

        return self._load_from_file(checkpoint_file)

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint metadata.
        """
        checkpoints_dir = self.state_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return []

        checkpoints = []
        for checkpoint_file in sorted(checkpoints_dir.glob("cp_*.json")):
            try:
                with open(checkpoint_file, encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints.append({
                    "checkpoint_id": checkpoint_file.stem,
                    "phase": data.get("current_phase"),
                    "prompt": data.get("prompt", "")[:100],
                    "created_at": data.get("updated_at"),
                    "reason": data.get("checkpoint_reason"),
                })
            except Exception as e:
                logger.warning("Failed to read checkpoint %s: %s", checkpoint_file, e)

        return checkpoints

    async def clear_state(self) -> None:
        """Clear all state (main file and backups)."""
        if self.state_file.exists():
            self.state_file.unlink()

        for backup in self.backup_dir.glob("state_*.json"):
            backup.unlink()

        logger.info("State cleared")

    def _create_backup(self) -> None:
        """Create a backup of the current state file."""
        if not self.state_file.exists():
            return

        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        backup_file = self.backup_dir / f"state_{timestamp}.json"

        # Copy current state to backup
        backup_file.write_bytes(self.state_file.read_bytes())

        # Prune old backups
        self._prune_backups()

    def _prune_backups(self) -> None:
        """Remove old backups, keeping only MAX_BACKUPS most recent."""
        backups = sorted(self.backup_dir.glob("state_*.json"), reverse=True)

        for old_backup in backups[self.MAX_BACKUPS:]:
            old_backup.unlink()
            logger.debug("Pruned old backup %s", old_backup.name)

    def _load_from_file(self, path: Path) -> WorkflowState:
        """Load state from a specific file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return WorkflowState.from_checkpoint_dict(data)
