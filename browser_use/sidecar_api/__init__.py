"""HTTP sidecar API exposing browser-use training & task execution.

Designed to be called from external backends (e.g. .NET) over HTTP.
The FastAPI app instance lives in `browser_use.sidecar_api.app:app`.
"""

from browser_use.sidecar_api.app import app

__all__ = ['app']
