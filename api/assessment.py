"""Assessment routes isolated from model training and inference."""
from typing import Any, Dict, Optional
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from evaluation.output_assessment import generate_assessment

router=APIRouter(prefix='/api/v1/assessment',tags=['Validation, vulnerability and reports'])


class AssessmentRequest(BaseModel):
    prediction: Dict[str,Any]
    exposure: Optional[Dict[str,Any]]=None
    source: Optional[Dict[str,Any]]=None
    use_llm: bool=False


@router.get('/status')
def assessment_status():
    return {'validation':'available','exposure_analysis':'requires supplied WGS84 GeoJSON',
        'structured_report':'available','local_llm_configured':bool(os.getenv('OLLAMA_MODEL','').strip()),
        'local_llm_model':os.getenv('OLLAMA_MODEL','').strip() or None,
        'note':'Configuration does not confirm that the local model/service is running.'}


@router.post('/generate')
def assessment_generate(body: AssessmentRequest):
    try:
        return generate_assessment(body.prediction,body.exposure,body.source,body.use_llm)
    except (ValueError,TypeError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
