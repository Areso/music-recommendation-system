#!/usr/bin/env python3
"""Interactive terminal client for the Music Recommendation API."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    Completer,
    Completion,
    ThreadedCompleter,
)
from prompt_toolkit.document import Document


DEFAULT_API_URL = "http://127.0.0.1:8000/api"
CONFIG_PATH = Path(__file__).resolve().with_name("cli_client.config")
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


def load_api_url() -> str:
    """Load the last selected API URL, falling back to the local server."""
    try:
        saved_url = CONFIG_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return DEFAULT_API_URL
    except OSError as exc:
        print(f"Warning: could not read {CONFIG_PATH.name}: {exc}")
        return DEFAULT_API_URL

    try:
        return normalize_api_url(saved_url)
    except ValueError as exc:
        print(f"Warning: ignoring invalid {CONFIG_PATH.name}: {exc}")
        return DEFAULT_API_URL


def save_api_url(url: str) -> None:
    """Persist the selected API URL for the next CLI session."""
    CONFIG_PATH.write_text(f"{url}\n", encoding="utf-8")


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


class CommandCompleter(Completer):
    """Complete commands and their remote tag/artist arguments."""

    commands = (
        "/connect",
        "/check",
        "/tag_search",
        "/similar_artists",
        "/find_similar_user",
        "/recommend_content",
        "/recommend_cf",
        "/help",
        "/quit",
        "/exit",
    )

    def __init__(self, client: ApiClient):
        self.tag_completer = TagCompleter(client)
        self.artist_completer = ArtistCompleter(client)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        for command, completer in (
            ("/tag_search", self.tag_completer),
            ("/similar_artists", self.artist_completer),
        ):
            prefix = f"{command} "
            if text.startswith(prefix):
                argument = text[len(prefix) :]
                if not argument.strip():
                    return
                argument_document = Document(
                    text=argument,
                    cursor_position=len(argument),
                )
                yield from completer.get_completions(
                    argument_document,
                    complete_event,
                )
                return

        if " " in text:
            return

        for command in self.commands:
            if command.startswith(text):
                yield Completion(command, start_position=-len(text))


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


def run_find_similar_user(
    client: ApiClient,
    session: PromptSession,
    initial_query: str = "",
) -> None:
    value = initial_query.strip()
    if not value:
        value = session.prompt("user ID> ").strip()
    if not value:
        print("Similar-user search cancelled.")
        return

    try:
        user_id = int(value)
    except ValueError:
        print("User ID must be an integer.")
        return

    data = client.get("similar_users", q=user_id)
    if data.get("error"):
        print(f"Error: {data['error']}")
        return

    print(
        f"\nUsers similar to user {data.get('query', user_id)} "
        f"({data.get('artist_count', '?')} listened artists):"
    )
    if not data.get("confident", True):
        print("Warning: the user has too little listening data for a reliable ranking.")

    results = data.get("results", [])
    if not results:
        print("No similar users found.")
        return

    print_table(
        ["User ID", "Score", "Artists", "Shared", "Shared artists"],
        [
            [
                result["user_id"],
                result["score"],
                result["artist_count"],
                result["shared_artist_count"],
                ", ".join(
                    artist["artist_name"]
                    for artist in result.get("shared_artists", [])
                ),
            ]
            for result in results
        ],
    )


def parse_user_id(
    session: PromptSession,
    initial_query: str,
    cancelled_message: str,
) -> int | None:
    value = initial_query.strip()
    if not value:
        value = session.prompt("user ID> ").strip()
    if not value:
        print(cancelled_message)
        return None

    try:
        return int(value)
    except ValueError:
        print("User ID must be an integer.")
        return None


def run_content_recommendations(
    client: ApiClient,
    session: PromptSession,
    initial_query: str = "",
) -> None:
    user_id = parse_user_id(
        session,
        initial_query,
        "Content recommendation cancelled.",
    )
    if user_id is None:
        return

    data = client.get("recommend_content", q=user_id)
    if data.get("error"):
        print(f"Error: {data['error']}")
        return

    print(
        f"\nContent recommendations for user {data.get('query', user_id)} "
        f"({data.get('covered_artist_count', '?')} of "
        f"{data.get('artist_count', '?')} listened artists covered):"
    )
    profile_tags = data.get("profile_tags", [])
    if profile_tags:
        print(f"Profile tags: {', '.join(profile_tags)}")
    if not data.get("confident", True):
        print("Warning: too few listened artists have content vectors for a reliable profile.")

    results = data.get("results", [])
    if not results:
        print("No content-based recommendations found.")
        return

    print_table(
        ["Rank", "Artist ID", "Artist", "Score", "Top tags"],
        [
            [
                rank,
                result["artist_id"],
                result["artist_name"],
                result["score"],
                ", ".join(result.get("top_tags", [])),
            ]
            for rank, result in enumerate(results, start=1)
        ],
    )


def run_cf_recommendations(
    client: ApiClient,
    session: PromptSession,
    initial_query: str = "",
) -> None:
    user_id = parse_user_id(
        session,
        initial_query,
        "Collaborative-filtering recommendation cancelled.",
    )
    if user_id is None:
        return

    data = client.get("recommend_cf", q=user_id)
    if data.get("error"):
        print(f"Error: {data['error']}")
        return

    print(
        f"\nCollaborative-filtering recommendations for user "
        f"{data.get('query', user_id)} using "
        f"{data.get('neighbors_used', '?')} neighbours:"
    )
    if "top_neighbor_similarity" in data:
        print(f"Closest-neighbour similarity: {data['top_neighbor_similarity']}")
    if not data.get("confident", True):
        print("Warning: the user has too little listening data for a reliable ranking.")

    results = data.get("results", [])
    if not results:
        print("No collaborative-filtering recommendations found.")
        return

    print_table(
        ["Rank", "Artist ID", "Artist", "CF score", "Neighbour support"],
        [
            [
                rank,
                result["artist_id"],
                result["artist_name"],
                result["score"],
                result["neighbor_support"],
            ]
            for rank, result in enumerate(results, start=1)
        ],
    )


HELP = """Commands:
  /connect <host|host:port|URL>  Set and check the API server (plain hosts use port 8000)
  /check                         Check whether the current API server is up
  /tag_search [tag]              Find artists by tag, with live autocomplete
  /similar_artists [name|ID]     Find similar artists, with live autocomplete
  /find_similar_user [user ID]   Find users with similar listening histories
  /recommend_content [user ID]   Recommend unlistened artists from the user's tag profile
  /recommend_cf [user ID]        Recommend unlistened artists from similar users
  /help                          Show this help
  /quit, /exit                   Exit
"""


def main() -> int:
    client = ApiClient(load_api_url())
    session: PromptSession = PromptSession()
    command_completer = ThreadedCompleter(CommandCompleter(client))

    print("Music Recommendation CLI")
    print(f"Current server: {client.base_url}")
    try:
        check_server(client)
    except ClientError as exc:
        print(f"DOWN — {client.base_url}")
        print(f"Error: {exc}")
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
                try:
                    save_api_url(client.base_url)
                except OSError as exc:
                    print(f"Warning: could not save {CONFIG_PATH.name}: {exc}")
                check_server(client)
            elif command == "/check":
                check_server(client)
            elif command == "/tag_search":
                run_tag_search(client, session, argument)
            elif command == "/similar_artists":
                run_similar_artists(client, session, argument)
            elif command == "/find_similar_user":
                run_find_similar_user(client, session, argument)
            elif command == "/recommend_content":
                run_content_recommendations(client, session, argument)
            elif command == "/recommend_cf":
                run_cf_recommendations(client, session, argument)
            elif command == "/help":
                print(HELP)
            elif command in {"/quit", "/exit", "/q"}:
                return 0
            else:
                print(f"Unknown command: {parts[0]}. Type /help.")
        except ClientError as exc:
            print(f"Error: {exc}")
        except KeyboardInterrupt:
            print("\nCancelled.")


if __name__ == "__main__":
    raise SystemExit(main())
