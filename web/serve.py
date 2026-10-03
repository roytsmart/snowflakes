"""
Serve the web app locally, as GitHub Pages serves it.

Build the kernel first, then run this from the root of the repository and
open http://localhost:8000::

    cargo build --release --target wasm32-unknown-unknown --manifest-path rust/Cargo.toml
    python web/serve.py

The kernel is served as ``snowflakes.wasm`` from Cargo's build directory,
which is ``rust/target`` unless ``CARGO_TARGET_DIR`` says otherwise.
"""

import functools
import http.server
import os
import pathlib
import sys

WEB = pathlib.Path(__file__).parent
TARGET = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", WEB.parent / "rust" / "target"))
WASM = TARGET / "wasm32-unknown-unknown" / "release" / "snowflakes.wasm"


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the app, and the kernel from where Cargo built it."""

    def translate_path(self, path: str) -> str:
        route = path.split("?", 1)[0].split("#", 1)[0]
        if route == "/snowflakes.wasm":
            return str(WASM)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    if not WASM.exists():
        sys.exit(f"Build the kernel first: {WASM} is missing.")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(Handler, directory=str(WEB))
    with http.server.ThreadingHTTPServer(("", port), handler) as server:
        print(f"Serving the app at http://localhost:{port}")
        server.serve_forever()
