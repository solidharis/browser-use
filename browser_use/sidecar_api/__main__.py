"""Run the sidecar with: `python -m browser_use.sidecar_api`"""

import uvicorn


def main() -> None:
	uvicorn.run('browser_use.sidecar_api.app:app', host='127.0.0.1', port=8765, reload=False)


if __name__ == '__main__':
	main()
