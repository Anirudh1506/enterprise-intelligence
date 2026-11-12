from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from app.api.routes.documents import router as documentRouter

app = FastAPI(
    title="Enterprise Intelligence Platform",
    description="Evidence-driven enterprise AI platform",
    version="0.1.0",
)

app.include_router(documentRouter, prefix = "/api/v1/documents", tags = ["Documents"])

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-intelligence-api",
    }