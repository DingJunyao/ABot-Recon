"""Serve the project-local pseudo-Gaussian viewer from the repository root."""

from __future__ import annotations

import argparse
import functools
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def viewer_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/tools/pseudo_gaussian_viewer.html"


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the project-local pseudo-Gaussian viewer."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the viewer URL in the default browser",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")

    root = repository_root()
    handler = functools.partial(ViewerRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/tools/pseudo_gaussian_viewer.html"
    print(f"Serving {root}")
    print(f"Viewer: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
