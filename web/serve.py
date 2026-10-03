"""
Serve the web app locally, as GitHub Pages serves it.

Run ``python web/serve.py`` from the root of the repository and open
http://localhost:8000. The package's files are served under ``snowflakes/``,
where the app's worker loads them into Pyodide.
"""

import functools
import http.server
import pathlib
import sys

WEB = pathlib.Path(__file__).parent
PACKAGE = WEB.parent / "snowflakes"


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the app, and the package beside it."""

    def translate_path(self, path: str) -> str:
        route = path.split("?", 1)[0].split("#", 1)[0]
        if route.startswith("/snowflakes/"):
            return str(PACKAGE / route.removeprefix("/snowflakes/"))
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(Handler, directory=str(WEB))
    with http.server.ThreadingHTTPServer(("", port), handler) as server:
        print(f"Serving the app at http://localhost:{port}")
        server.serve_forever()
