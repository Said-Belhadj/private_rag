"""FastAPI application entry point for the Private RAG REST API."""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    """Health-check endpoint.

    Returns:
        A JSON object with a ``message`` key confirming the API is running.
    """
    return {"message": "Hello World"}
