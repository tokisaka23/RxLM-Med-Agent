import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# Import core reasoning modules
from agent.medical_calc import calculate_derived_metrics
from agent.reflection_logic import run_progressive_pruning

# ======================
# Configuration & Logging
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rxlmed-api")

# Ensure report output directory exists
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Optional: Load quantized model
QUANTIZED_MODEL_PATH = os.getenv("QUANTIZED_MODEL_PATH", "./final_agent_int4")
USE_QUANTIZED = os.path.exists(QUANTIZED_MODEL_PATH)

if USE_QUANTIZED:
    logger.info(f"✅ INT4 quantized model detected at {QUANTIZED_MODEL_PATH}")
else:
    logger.warning("⚠️  Using full-precision model. Set QUANTIZED_MODEL_PATH for INT4.")

# ======================
# Pydantic Models
# ======================

class LabItem(BaseModel):
    abbreviation: str
    value: float
    unit: str = ""

class LabInput(BaseModel):
    gender: str = Field(..., pattern="^(M|F)$", description="M for male, F for female")
    age: int = Field(..., ge=0, le=120)
    lab_items: List[LabItem]

class PatientContext(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    exclude_history: List[str] = Field(default_factory=list)

class AnalysisRequest(BaseModel):
    lab_data: LabInput
    patient_context: PatientContext

class TLPResponse(BaseModel):
    request_id: str
    timestamp: str
    tlp_status: str = Field(..., pattern="^(GREEN|YELLOW|RED)$")
    clinical_conclusion: str
    risk_level_assessment: str
    derived_metrics: Dict[str, Any]
    reasoning_trace: List[str]
    background_report_path: str = ""

# ======================
# Background Task: Full Report Generation
# ======================

def generate_detailed_report(
    request_id: str,
    lab_data: dict,
    patient_context: dict,
    reasoning_result: dict
):
    """
    🌟 Background dual-channel rendering:
    - Main channel: Immediate TLP response (<500ms)
    - Background channel: Full clinical report (PDF/Markdown, ~2s)
    """
    try:
        report_data = {
            "request_id": request_id,
            "generated_at": datetime.now().isoformat(),
            "lab_data": lab_data,
            "patient_context": patient_context,
            "reasoning_chain": reasoning_result["reasoning_trace"],
            "final_conclusion": reasoning_result["clinical_conclusion"],
            "risk_level": reasoning_result["risk_level_assessment"],
            "derived_metrics": reasoning_result["derived_metrics"]
        }

        report_path = REPORTS_DIR / f"report_{request_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 Detailed report saved: {report_path}")

        # Optional: Trigger PDF generation, email notification, etc.
        # render_pdf_from_json(report_path)

    except Exception as e:
        logger.error(f"Failed to generate background report: {e}")

# ======================
# FastAPI App
# ======================

app = FastAPI(
    title="RxLM-Med Clinical Reasoning Agent",
    description="System 2 medical reasoning engine with Traffic Light Protocol compliance",
    version="1.0.0",
    contact={
        "name": "RxLM-Med Team",
        "email": "support@rxlmed.ai"
    }
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Kubernetes / monitoring health endpoint"""
    return {"status": "healthy", "quantized_model_loaded": USE_QUANTIZED}

@app.post(
    "/v1/analyze",
    response_model=TLPResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze lab report with System 2 reasoning",
    description="""
    Executes a deterministic calculation → progressive pruning loop → TLP-compliant output.
    Returns immediately with traffic light status; detailed report generated in background.
    """
)
@traceable(run_type="llm")  # LangSmith: marks this as an 'LLM'-type run (even if rule-based)
async def analyze_medical_report(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    🌟 Asynchronous dual-channel clinical analysis endpoint.
    
    Main Channel (Fast):
      - Returns TLP status (GREEN/YELLOW/RED) within <500ms
      - Contains safety-critical conclusion only
    
    Background Channel (Slow):
      - Generates full audit trail, metrics, and reasoning chain
      - Saved to /reports/ for clinician review
    """
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    logger.info(f"📥 Received analysis request: {request_id}")

    try:
        # === Step 1: Deterministic Calculation (The Guardian) ===
        extended_data = calculate_derived_metrics(
            gender=request.lab_data.gender,
            age=request.lab_data.age,
            lab_items=[item.dict() for item in request.lab_data.lab_items]
        )

        # === Step 2: System 2 Reflection Loop (The Critic + Refiner) ===
        await asyncio.sleep(0.05)  # Non-blocking wait
        reasoning_result = run_progressive_pruning(
            extended_lab_data=extended_data,
            patient_context=request.patient_context.dict()
        )

        # === Step 3: Construct TLP Response ===
        tlp_status = reasoning_result["risk_level_assessment"]
        response = TLPResponse(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            tlp_status=tlp_status,
            clinical_conclusion=reasoning_result["clinical_conclusion"],
            risk_level_assessment=tlp_status,
            derived_metrics=reasoning_result["derived_metrics"],
            reasoning_trace=reasoning_result["reasoning_trace"],
            background_report_path=f"/reports/report_{request_id}.json"
        )

        # === Step 4: Enqueue Background Report Generation ===
        background_tasks.add_task(
            generate_detailed_report,
            request_id=request_id,
            lab_data=request.lab_data.dict(),
            patient_context=request.patient_context.dict(),
            reasoning_result=reasoning_result
        )

        # === Step 5: Log to LangSmith (optional manual span) ===
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.end(outputs={"tlp_status": tlp_status, "conclusion": reasoning_result["clinical_conclusion"]})

        logger.info(f"📤 Responded with TLP-{tlp_status}: {request_id}")
        return response

    except ValueError as ve:
        logger.error(f"ValueHandling Input error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(ve)}"
        )
    except Exception as e:
        logger.exception(f"💥 Unexpected error in request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal reasoning engine error"
        )