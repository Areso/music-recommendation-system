#!/usr/bin/env python3
"""Interactive terminal client for the Music Recommendation API."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    Completer,
    Completion,
    ThreadedCompleter,
    WordCompleter,
)


DEFAULT_API_URL = "http://127.0.0.1:8000/api"
REQUEST_TIMEOUT = 5
ARTIST_SELECTION_RE = re.compile(r"^.+ \[(?P<artist_id>[^\[\]]+)\]$")


class ClientError(Exception):
    """A connection, HTTP, or response error from the API."""


def normalize_api_url(value: str) -> str:
    """Turn a host, server URL, or API URL into a normalized API base URL."""
    raw = value.strip()
    if not raw:
        raise ValueError("Address cannot be empty.")

    has_scheme = "://" in raw
    candidate = raw if has_scheme else f"http://{raw}"

    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid address: {exc}") from exc

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// addresses are supported.")
    if not parsed.hostname:
        raise ValueError("Address must include an IP address or DNS name.")
    if parsed.username or parsed.password:
        raise ValueError("Credentials in the URL are not supported.")
    if parsed.query or parsed.fragment:
        raise ValueError("Query strings and fragments are not valid server addresses.")

    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    if port is not None:
        netloc = f"{host}:{port}"
    elif not has_scheme:
        netloc = f"{host}:8000"
    else:
        netloc = host

    path = parsed.path.rstrip("/")
    if not path.endswith("/api"):
        path = f"{path}/api"

    return urlunsplit((parsed.scheme, netloc, path, "", ""))


@dataclass
class ApiClient:
    base_url: str = DEFAULT_API_URL
    timeout: float = REQUEST_TIMEOUT

    def __post_init__(self) -> None:
        self._completion_cache: dict[tuple[str, str], list[Any]] = {}

    def set_base_url(self, value: str) -> None:
        self.base_url = normalize_api_url(value)
        self._completion_cache.clear()

    def get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        query = urlencode(params)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query:
            url = f"{url}?{query}"

        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "music-recommender-cli"},
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if detail:
                try:
                    payload = json.loads(detail)
                    detail = payload.get("detail", detail)
                except json.JSONDecodeError:
                    pass
            suffix = f": {detail}" if detail else ""
            raise ClientError(f"HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ClientError(f"Cannot reach {self.base_url}: {reason}") from exc
        except TimeoutError as exc:
            raise ClientError(f"Request timed out after {self.timeout:g} seconds.") from exc

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ClientError("Server returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise ClientError("Server returned an unexpected response.")
        return payload

    def suggestions(self, endpoint: str, query: str) -> list[Any]:
        key = (endpoint, query.casefold())
        if key not in self._completion_cache:
            payload = self.get(endpoint, q=query, limit=10)
            suggestions = payload.get("suggestions", [])
            self._completion_cache[key] = (
                suggestions if isinstance(suggestions, list) else []
            )
        return self._completion_cache[key]


class TagCompleter(Completer):
    def __init__(self, client: ApiClient):
        self.client = client

    def get_completions(self, document, complete_event):
        query = document.text.strip()
        if not query:
            return

        try:
            suggestions = self.client.suggestions("autocomplete", query)
        except ClientError:
            return

        for tag in suggestions:
            if isinstance(tag, str):
                yield Completion(tag, start_position=-len(document.text))


class ArtistCompleter(Completer):
    def __init__(self, client: ApiClient):
        self.client = client

    def get_completions(self, document, complete_event):
        query = document.text.strip()
        if not query:
            return

        try:
            suggestions = self.client.suggestions("autocomplete_artist", query)
        except ClientError:
            return

        for artist in suggestions:
            if not isinstance(artist, dict):
                continue
            artist_id = artist.get("artist_id")
            artist_name = artist.get("artist_name")
            if artist_id is None or not isinstance(artist_name, str):
                continue

            yield Completion(
                f"{artist_name} [{artist_id}]",
                start_position=-len(document.text),
                display=artist_name,
                display_meta=f"ID {artist_id}",
            )


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in text_rows))
        for i, header in enumerate(headers)
    ]

    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def check_server(client: ApiClient) -> None:
    data = client.get("healthcheck")
    if data.get("status") != "ok":
        raise ClientError(f"Server returned unhealthy status: {data.get('status')}")

    tfidf_shape = data.get("tfidf_matrix", ["?", "?"])
    user_artist_shape = data.get("user_artist_matrix", ["?", "?"])
    print(f"UP — {client.base_url}")
    print(
        f"{data.get('artists', '?')} artists, {data.get('tags', '?')} tags; "
        f"TF-IDF {tfidf_shape}, user-artist {user_artist_shape}"
    )


def run_tag_search(
    client: ApiClient,
    session: PromptSession,
    initial_query: str = "",
) -> None:
    query = initial_query.strip()
    if not query:
        query = session.prompt(
            "tag> ",
            completer=ThreadedCompleter(TagCompleter(client)),
            complete_while_typing=True,
        ).strip()
    if not query:
        print("Tag search cancelled.")
        return

    data = client.get("search", q=query)
    results = data.get("results", [])
    if not results:
        print(f'No artists matched "{query}".')
        return

    print(f'\nTop matches for "{data.get("query", query)}":')
    print_table(
        ["Artist ID", "Artist", "Score"],
        [
            [result["artist_id"], result["artist_name"], result["score"]]
            for result in results
        ],
    )


def resolve_artist_id(client: ApiClient, value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return value

    match = ARTIST_SELECTION_RE.match(value)
    if match:
        return match.group("artist_id")

    suggestions = client.suggestions("autocomplete_artist", value)
    exact = [
        artist
        for artist in suggestions
        if isinstance(artist, dict)
        and str(artist.get("artist_name", "")).casefold() == value.casefold()
    ]
    if len(exact) == 1:
        return str(exact[0]["artist_id"])

    print("Select an autocomplete suggestion, or enter an artist ID.")
    return None


def run_similar_artists(
    client: ApiClient,
    session: PromptSession,
    initial_query: str = "",
) -> None:
    value = initial_query.strip()
    if not value:
        value = session.prompt(
            "artist> ",
            completer=ThreadedCompleter(ArtistCompleter(client)),
            complete_while_typing=True,
        ).strip()

    artist_id = resolve_artist_id(client, value)
    if artist_id is None:
        return

    data = client.get("similar_artists", q=artist_id)
    if data.get("error"):
        print(f"Error: {data['error']}")
        return

    artist_name = data.get("artist_name", artist_id)
    print(
        f'\nArtists similar to "{artist_name}" '
        f"({data.get('tag_count', '?')} seed tags):"
    )
    if not data.get("confident", True):
        print("Warning: the seed has too few tags for a reliable ranking.")

    results = data.get("results", [])
    if not results:
        print("No similar artists found.")
        return

    print_table(
        ["Artist ID", "Artist", "Score", "Tags", "Shared tags"],
        [
            [
                result["artist_id"],
                result["artist_name"],
                result["score"],
                result["tag_count"],
                ", ".join(result.get("shared_tags", [])),
            ]
            for result in results
        ],
    )


HELP = """Commands:
  /connect <host|host:port|URL>  Set the API server (plain hosts use port 8000)
  /check                         Check whether the current API server is up
  /tag_search [tag]              Find artists by tag, with live autocomplete
  /similar_artists [name|ID]     Find similar artists, with live autocomplete
  /help                          Show this help
  /quit                          Exit
"""


def main() -> int:
    client = ApiClient()
    session: PromptSession = PromptSession()
    command_completer = WordCompleter(
        ["/connect", "/check", "/tag_search", "/similar_artists", "/help", "/quit"]
    )

    print("Music Recommendation CLI")
    print(f"Current server: {client.base_url}")
    print("Type /help for commands.")

    while True:
        try:
            line = session.prompt(
                "music> ",
                completer=command_completer,
                complete_while_typing=True,
            ).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"Invalid command: {exc}")
            continue

        command = parts[0].lower()
        argument = " ".join(parts[1:])

        try:
            if command == "/connect":
                if not argument:
                    print("Usage: /connect <host|host:port|URL>")
                    continue
                client.set_base_url(argument)
                print(f"Current server: {client.base_url}")
            elif command == "/check":
                check_server(client)
            elif command == "/tag_search":
                run_tag_search(client, session, argument)
            elif command == "/similar_artists":
                run_similar_artists(client, session, argument)
            elif command == "/help":
                print(HELP)
            elif command in {"/quit", "/exit"}:
                return 0
            else:
                print(f"Unknown command: {parts[0]}. Type /help.")
        except ClientError as exc:
            print(f"Error: {exc}")
        except KeyboardInterrupt:
            print("\nCancelled.")


if __name__ == "__main__":
    raise SystemExit(main())
