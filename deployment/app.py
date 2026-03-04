from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from langsmith import traceable  # LangSmith 集成

from agent.medical_cal import calculate_derived_metrics
from agent.reflection_logic import run_progressive_pruning

app = FastAPI(title="RxLM-Med Agent API", version="1.0")

class LabInput(BaseModel):
    gender: str
    age: int
    lab_items: List[dict]

class PatientContext(BaseModel):
    symptoms: List[str]
    exclude_history: List[str]

class AnalysisRequest(BaseModel):
    lab_data: LabInput
    patient_context: PatientContext

class TLPResponse(BaseModel):
    reasoning_trace: List[str]
    clinical_conclusion: str
    risk_level_assessment: str
    derived_metrics: dict
    tlp_status: str  # "GREEN", "YELLOW", "RED"

@app.post("/v1/analyze", response_model=TLPResponse)
@traceable  # LangSmith 自动追踪此函数
async def analyze_medical_report(request: AnalysisRequest):
    """
    异步执行 System 2 渐进式剪枝循环
    """
    try:
        # Step 1: 数学计算
        extended_data = calculate_derived_metrics(
            gender=request.lab_data.gender,
            age=request.lab_data.age,
            lab_items=request.lab_data.lab_items
        )

        # Step 2: 反思推理（模拟耗时操作）
        await asyncio.sleep(0.1)  # 模拟 I/O
        result = run_progressive_pruning(extended_data, request.patient_context.dict())

        # Step 3: 构建 TLP 响应
        response = TLPResponse(
            reasoning_trace=result["reasoning_trace"],
            clinical_conclusion=result["clinical_conclusion"],
            risk_level_assessment=result["risk_level_assessment"],
            derived_metrics=result["derived_metrics"],
            tlp_status=result["risk_level_assessment"]  # 直接映射
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")