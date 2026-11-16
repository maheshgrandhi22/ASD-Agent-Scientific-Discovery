"""Persistent storage for resumable scientific research sessions."""

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ResearchSessionMemory:
    """Save, retrieve, and update independent research sessions."""

    def __init__(self, storage_directory="data/research_sessions"):
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_id(session_id):
        if not isinstance(session_id, str) or not re.fullmatch(
            r"[a-f0-9]{32}", session_id
        ):
            raise ValueError("Invalid research session ID.")
        return session_id

    def _path(self, session_id):
        return self.storage_directory / (
            self._validate_id(session_id) + ".json"
        )

    def create_session(self, research_question, initial_data=None):
        if not isinstance(research_question, str) or not research_question.strip():
            raise ValueError("Research question must be non-empty.")

        if initial_data is not None and not isinstance(initial_data, dict):
            raise ValueError("Initial session data must be a dictionary.")

        now = self._now()
        session_id = uuid.uuid4().hex
        session = {
            "session_id": session_id,
            "research_question": research_question.strip(),
            "created_at": now,
            "updated_at": now,
            "status": "in_progress",
            "workflow_stage": "started",
            "scientific_validation": "not_performed",
            "verified_literature": [],
            "candidate_hypotheses": [],
            "experiment_results": {},
            "progress": {},
            "events": [],
        }

        if initial_data:
            protected = {
                "session_id", "created_at", "updated_at",
                "research_question",
            }
            for key, value in initial_data.items():
                if key not in protected:
                    session[key] = value

        self._write(session)
        return session

    def _write(self, session):
        path = self._path(session["session_id"])
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.storage_directory,
                prefix=".session_",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                json.dump(session, file, indent=2, ensure_ascii=False)
                file.write("\n")
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def get_session(self, session_id):
        path = self._path(session_id)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as file:
            session = json.load(file)
        if (
            not isinstance(session, dict)
            or session.get("session_id") != session_id
        ):
            raise ValueError("Stored research session is invalid.")
        return session

    def update_session(self, session_id, updates):
        if not isinstance(updates, dict):
            raise ValueError("Session updates must be a dictionary.")

        session = self.get_session(session_id)
        if session is None:
            raise KeyError("Research session was not found.")

        protected = {
            "session_id", "created_at", "updated_at",
            "research_question",
        }
        for key, value in updates.items():
            if key not in protected:
                session[key] = value

        session["updated_at"] = self._now()
        self._write(session)
        return session

    def add_event(self, session_id, event, details=None):
        if not isinstance(event, str) or not event.strip():
            raise ValueError("Event description must be non-empty.")
        if details is not None and not isinstance(details, dict):
            raise ValueError("Event details must be a dictionary.")

        session = self.get_session(session_id)
        if session is None:
            raise KeyError("Research session was not found.")

        session["events"].append({
            "timestamp": self._now(),
            "event": event.strip(),
            "details": details or {},
        })
        session["updated_at"] = self._now()
        self._write(session)
        return session

    def list_sessions(self):
        sessions = []
        for path in self.storage_directory.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as file:
                    item = json.load(file)
                if isinstance(item, dict) and isinstance(
                    item.get("session_id"), str
                ):
                    sessions.append({
                        "session_id": item["session_id"],
                        "research_question": item.get("research_question", ""),
                        "status": item.get("status", "unknown"),
                        "created_at": item.get("created_at"),
                        "updated_at": item.get("updated_at"),
                        "workflow_stage": item.get("workflow_stage", "unknown"),
                    })
            except (OSError, json.JSONDecodeError):
                continue

        return sorted(
            sessions,
            key=lambda item: item.get("updated_at") or "",
            reverse=True,
        )
