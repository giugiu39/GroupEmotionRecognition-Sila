from fastapi import FastAPI
from routers import emonodes, app_routes

app = FastAPI(title="Group Emotion Recognition API", version="1.0.0")

app.include_router(emonodes.router)
app.include_router(app_routes.router)
