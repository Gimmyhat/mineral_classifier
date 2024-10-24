from fastapi import APIRouter
from app.core.models import BatchMineralRequest, BatchMineralResponse, MineralClassification
from app.services.classifier import classifier

router = APIRouter()


@router.post("/classify-batch", response_model=BatchMineralResponse)
async def classify_batch(request: BatchMineralRequest):
    results = [
        MineralClassification(**classifier.classify(mineral))
        for mineral in request.minerals
    ]
    return BatchMineralResponse(results=results)
