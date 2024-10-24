from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api import routes as api_routes
from app.web import routes as web_routes

app = FastAPI(title=settings.PROJECT_NAME)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключаем роутеры
app.include_router(api_routes.router, prefix=settings.API_V1_STR)
app.include_router(web_routes.router)
