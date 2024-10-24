from pydantic import BaseModel
from typing import List


class MineralRequest(BaseModel):
    name: str


class BatchMineralRequest(BaseModel):
    minerals: List[str]


class MineralClassification(BaseModel):
    mineral_name: str
    pi_group_is_nedra: str
    pi_name_gbz_tbz: str
    normalized_name_for_display: str
    pi_measurement_unit: str
    pi_measurement_unit_alternative: str


class BatchMineralResponse(BaseModel):
    results: List[MineralClassification]
