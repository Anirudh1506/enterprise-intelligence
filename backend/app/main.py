from fastapi import FastAPI
app = FastAPI(
    title="Enterprise Intelligence Platform",
    description="Evidence-driven enterprise AI platform",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-intelligence-api",
    }