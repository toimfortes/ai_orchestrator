"""
Gmail API adapter for OAuth authentication and email export.
Exports raw emails in RFC 2822 format for forensic preservation.
"""

import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Iterator, Callable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import get_settings, GmailConfig
from .models import EmailRecord, EmailHeader, EmailBody, AttachmentInfo

logger = logging.getLogger(__name__)


class GmailAdapter:
    """Adapter for Gmail API operations."""

    def __init__(
        self,
        account: str,
        credentials_file: Optional[Path] = None,
        token_dir: Optional[Path] = None,
    ):
        """
        Initialize Gmail adapter for a specific account.

        Args:
            account: Email account to connect to
            credentials_file: Path to OAuth credentials.json
            token_dir: Directory to store token files
        """
        self.account = account
        self.settings = get_settings()
        self.gmail_config = self.settings.gmail

        self.credentials_file = credentials_file or Path(self.gmail_config.credentials_file)
        self.token_dir = token_dir or self.settings.output_dir / "tokens"
        self.token_dir.mkdir(parents=True, exist_ok=True)

        self.token_file = self.token_dir / self.gmail_config.token_file_pattern.format(
            account=account.replace("@", "_at_").replace(".", "_")
        )

        self._service = None
        self._credentials = None

    def authenticate(self) -> bool:
        """
        Authenticate with Gmail API using OAuth2.

        Returns:
            True if authentication successful
        """
        logger.info(f"Authenticating Gmail account: {self.account}")

        try:
            # Try to load existing token
            if self.token_file.exists():
                self._credentials = Credentials.from_authorized_user_file(
                    str(self.token_file),
                    self.gmail_config.scopes,
                )
                logger.info(f"Loaded existing token from {self.token_file}")

            # Refresh or get new credentials
            if not self._credentials or not self._credentials.valid:
                if self._credentials and self._credentials.expired and self._credentials.refresh_token:
                    logger.info("Refreshing expired credentials")
                    self._credentials.refresh(Request())
                else:
                    if not self.credentials_file.exists():
                        raise FileNotFoundError(
                            f"OAuth credentials file not found: {self.credentials_file}. "
                            "Download from Google Cloud Console."
                        )

                    logger.info("Starting OAuth flow - browser will open")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_file),
                        self.gmail_config.scopes,
                    )
                    self._credentials = flow.run_local_server(port=0)

                # Save token for future use
                self.token_file.write_text(self._credentials.to_json(), encoding="utf-8")
                logger.info(f"Saved token to {self.token_file}")

            # Build service
            self._service = build("gmail", "v1", credentials=self._credentials)
            logger.info(f"Gmail API service initialized for {self.account}")
            return True

        except Exception as e:
            logger.error(f"Authentication failed for {self.account}: {e}", exc_info=True)
            raise

    @property
    def service(self):
        """Get Gmail API service, authenticating if needed."""
        if self._service is None:
            self.authenticate()
        return self._service

    def build_search_query(self) -> str:
        """Build Gmail search query for target correspondents."""
        return self.settings.build_gmail_query()

    def search_messages(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[str]:
        """
        Search for messages matching query.

        Args:
            query: Gmail search query (default: target correspondents)
            max_results: Maximum number of results

        Returns:
            List of message IDs
        """
        if query is None:
            query = self.build_search_query()

        max_results = max_results or self.gmail_config.max_results_per_query

        logger.info(f"Searching messages for {self.account} with query: {query[:100]}...")

        message_ids = []
        page_token = None

        try:
            while len(message_ids) < max_results:
                results = self.service.users().messages().list(
                    userId="me",
                    q=query,
                    maxResults=min(self.gmail_config.batch_size, max_results - len(message_ids)),
                    pageToken=page_token,
                ).execute()

                messages = results.get("messages", [])
                if not messages:
                    break

                message_ids.extend(m["id"] for m in messages)
                logger.debug(f"Found {len(messages)} messages, total: {len(message_ids)}")

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

            logger.info(f"Found {len(message_ids)} matching messages for {self.account}")
            return message_ids

        except HttpError as e:
            logger.error(f"Gmail API error searching messages: {e}", exc_info=True)
            raise

    def get_message_raw(self, message_id: str) -> bytes:
        """
        Get raw email message in RFC 2822 format.

        Args:
            message_id: Gmail message ID

        Returns:
            Raw email bytes
        """
        try:
            message = self.service.users().messages().get(
                userId="me",
                id=message_id,
                format="raw",
            ).execute()

            raw_data = message.get("raw", "")
            return base64.urlsafe_b64decode(raw_data)

        except HttpError as e:
            logger.error(f"Failed to get raw message {message_id}: {e}", exc_info=True)
            raise

    def get_message_metadata(self, message_id: str) -> dict:
        """
        Get message metadata (headers, thread info).

        Args:
            message_id: Gmail message ID

        Returns:
            Message metadata dict
        """
        try:
            message = self.service.users().messages().get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=[
                    "From", "To", "Cc", "Bcc", "Subject", "Date",
                    "Message-ID", "In-Reply-To", "References",
                ],
            ).execute()

            return {
                "id": message["id"],
                "threadId": message["threadId"],
                "internalDate": int(message.get("internalDate", 0)),
                "headers": {
                    h["name"]: h["value"]
                    for h in message.get("payload", {}).get("headers", [])
                },
            }

        except HttpError as e:
            logger.error(f"Failed to get message metadata {message_id}: {e}", exc_info=True)
            raise

    def export_messages(
        self,
        output_dir: Path,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[tuple[str, bytes, dict]]:
        """
        Export messages to files.

        Args:
            output_dir: Directory to save emails
            query: Search query
            max_results: Maximum messages to export
            progress_callback: Callback(current, total) for progress updates

        Yields:
            Tuple of (message_id, raw_bytes, metadata)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        message_ids = self.search_messages(query, max_results)
        total = len(message_ids)

        logger.info(f"Exporting {total} messages from {self.account}")

        for idx, msg_id in enumerate(message_ids):
            try:
                raw_bytes = self.get_message_raw(msg_id)
                metadata = self.get_message_metadata(msg_id)

                if progress_callback:
                    progress_callback(idx + 1, total)

                yield msg_id, raw_bytes, metadata

            except Exception as e:
                logger.warning(f"Failed to export message {msg_id}: {e}")
                continue

    def close(self):
        """Close the Gmail service connection."""
        self._service = None
        self._credentials = None


def create_gmail_adapters(
    accounts: Optional[List[str]] = None,
    credentials_file: Optional[Path] = None,
) -> dict[str, GmailAdapter]:
    """
    Create Gmail adapters for multiple accounts.

    Args:
        accounts: List of email accounts (default from settings)
        credentials_file: OAuth credentials file

    Returns:
        Dict mapping account to adapter
    """
    settings = get_settings()
    accounts = accounts or settings.email_accounts

    adapters = {}
    for account in accounts:
        adapters[account] = GmailAdapter(
            account=account,
            credentials_file=credentials_file,
        )

    return adapters
