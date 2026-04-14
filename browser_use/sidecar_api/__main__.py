import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "browser_use.sidecar_api.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )
