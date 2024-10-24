from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from app.services.classifier import classifier

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request}
    )


@router.post("/classify")
async def classify_single(request: Request, mineral_name: str = Form(...)):
    result = classifier.classify(mineral_name)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result
        }
    )
