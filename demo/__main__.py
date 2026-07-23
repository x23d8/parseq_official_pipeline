"""Portable command-line entrypoint for the web demo."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Serve one inference worker using host-provided bind settings."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(
        "demo.app:app",
        host=host,
        port=port,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "*"),
    )


if __name__ == "__main__":
    main()
