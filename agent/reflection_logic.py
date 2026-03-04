# reflection_logic.py
import json
import os
from typing import Dict, List, Any

# ========== 模拟证据库（简化版）==========
EVIDENCE_DB = {
    "high_anion_gap_acidosis": {
        "causes": ["乳酸酸中毒", "酮症酸中毒", "肾衰", "中毒（甲醇、乙二醇）"],
        "text": "高阴离子间隙代谢性酸中毒常见于乳酸酸中毒、糖尿病酮症酸中毒、急慢性肾衰或毒物摄入。"
    },
    "lactic_acidosis": {
        "text": "乳酸 >4 mmol/L 提示 A型（组织缺氧）或 B型（非缺氧）乳酸酸中毒，需结合病史判断。"
    }
}

# ========== 模拟 The Drafter（故意引入幻觉）==========
def draft_initial_hypothesis(extended_data: Dict, patient_ctx: Dict) -> Dict:
    """
    生成初始草稿，可能包含无证据支持的推论（模拟 LLM 幻觉）
    """
    draft = {
        "hypotheses": [],
        "conclusion": "",
        "risk_level": "GREEN"
    }

    flags = extended_data.get("flags", {})
    metrics = extended_data.get("derived_metrics", {})
    symptoms = patient_ctx.get("symptoms", [])

    # 初始假设（可能错误）
    hypotheses = []

    if flags.get("AG_high"):
        hypotheses.append("患者存在高阴离子间隙代谢性酸中毒，可能由糖尿病酮症酸中毒引起。")  # 无血糖/尿酮证据！
    
    if flags.get("Lactate_critical"):
        hypotheses.append("乳酸显著升高，强烈提示感染性休克。")  # 无感染证据！

    if flags.get("PaO2_FiO2_abnormal"):
        hypotheses.append("氧合指数降低，符合急性呼吸窘迫综合征（ARDS）。")

    # 风险等级初判
    if flags.get("Lactate_critical") or flags.get("Hyperkalemia_critical"):
        draft["risk_level"] = "RED"
    elif flags:
        draft["risk_level"] = "YELLOW"
    else:
        draft["risk_level"] = "GREEN"

    draft["hypotheses"] = hypotheses
    draft["conclusion"] = "综合判断为多系统功能障碍，建议立即 ICU 会诊。"  # ⚠️ 过度推断
    return draft

# ========== 模拟 The Critic（规则化审查）==========
def critic_review(draft: Dict, extended_data: Dict, patient_ctx: Dict) -> Dict:
    """
    审查草稿是否违反事实或证据
    返回 { "passed": bool, "issues": [str] }
    """
    issues = []
    flags = extended_data.get("flags", {})
    metrics = extended_data.get("derived_metrics", {})
    lab_dict = {item["abbreviation"]: item["value"] for item in extended_data.get("original_lab", [])}

    # Rule 1: 是否忽略危急值？
    if flags.get("Lactate_critical") and "乳酸" not in str(draft):
        issues.append("未提及危急值：乳酸显著升高")

    # Rule 2: 是否做出无证据支持的病因推断？
    if "糖尿病酮症酸中毒" in str(draft):
        glu = lab_dict.get("Glu")
        if glu is None or glu < 11.0:  # 无高血糖证据
            issues.append("声称'糖尿病酮症酸中毒'，但血糖未达诊断标准且无尿酮证据")

    if "感染性休克" in str(draft):
        # 证据库中无感染相关证据（如 WBC、PCT、发热等）
        if "发热" not in patient_ctx.get("symptoms", []) and lab_dict.get("WBC") is None:
            issues.append("声称'感染性休克'，但缺乏感染证据（症状或白细胞）")

    # Rule 3: 结论是否与计算结果矛盾？
    if "ARDS" in draft.get("conclusion", ""):
        pao2_fio2 = metrics.get("PaO2_FiO2_estimated")
        if pao2_fio2 and pao2_fio2 > 200:
            issues.append("氧合指数未达 ARDS 诊断标准（需<200）")

    return {
        "passed": len(issues) == 0,
        "issues": issues
    }

# ========== 模拟 The Refiner（按 retry 策略修正）==========
def refine_draft(draft: Dict, critique: Dict, retry_count: int) -> Dict:
    """
    根据 Critic 反馈和重试次数修正草稿
    """
    new_draft = draft.copy()
    issues = critique["issues"]

    if retry_count == 1:
        # Attempt Repair: 替换病因为更宽泛描述
        text = new_draft["conclusion"]
        hypotheses = [h for h in new_draft["hypotheses"]]
        
        if any("糖尿病酮症酸中毒" in h for h in hypotheses):
            hypotheses = [h.replace("糖尿病酮症酸中毒", "代谢性酸中毒（病因待查）") for h in hypotheses]
        if any("感染性休克" in h for h in hypotheses):
            hypotheses = [h.replace("感染性休克", "乳酸酸中毒（原因待明确）") for h in hypotheses]
        
        new_draft["hypotheses"] = hypotheses
        new_draft["conclusion"] = "存在高阴离子间隙代谢性酸中毒合并乳酸升高，具体病因需进一步检查。"

    elif retry_count == 2:
        # Soft Pruning: 删除所有病因推测
        new_draft["hypotheses"] = ["存在高阴离子间隙代谢性酸中毒。", "乳酸显著升高（>4 mmol/L）。"]
        new_draft["conclusion"] = "实验室提示严重代谢性紊乱，建议紧急评估。"

    elif retry_count == 3:
        # Hard Pruning: 仅保留事实
        facts = []
        if draft.get("risk_level") == "RED":
            facts.append("检测到危急值：乳酸 >4.0 mmol/L。")
        if extended_data := globals().get("extended_data_global"):
            if extended_data["flags"].get("AG_high"):
                ag = extended_data["derived_metrics"]["Anion_Gap"]
                facts.append(f"阴离子间隙升高至 {ag} mmol/L（参考范围 10-20）。")
            if extended_data["flags"].get("PaO2_FiO2_abnormal"):
                pao2 = extended_data["derived_metrics"]["PaO2_FiO2_estimated"]
                facts.append(f"氧合指数估算为 {pao2}，低于正常（>400）。")
        new_draft["hypotheses"] = facts
        new_draft["conclusion"] = "上述异常指标需临床紧急干预。"
        new_draft["risk_level"] = "RED"  # 危急值存在，强制 RED

    return new_draft

# ========== 主循环：System 2 渐进式剪枝 ==========
def run_progressive_pruning(
    extended_lab_data: Dict,
    patient_context: Dict,
    max_retries: int = 3
) -> Dict:
    """
    执行 System 2 反思循环
    """
    global extended_data_global
    extended_data_global = extended_lab_data  # 供 Hard Pruning 使用

    reasoning_trace = []
    current_draft = draft_initial_hypothesis(extended_lab_data, patient_context)
    reasoning_trace.append("Initial hypothesis: " + "; ".join(current_draft["hypotheses"]))

    for retry in range(1, max_retries + 1):
        critique = critic_review(current_draft, extended_lab_data, patient_context)
        if critique["passed"]:
            reasoning_trace.append("Critic review PASSED.")
            break

        reasoning_trace.append(f"Correction (Retry {retry}): " + "; ".join(critique["issues"]))
        current_draft = refine_draft(current_draft, critique, retry)

        if retry == max_retries:
            reasoning_trace.append("Hard pruning applied. Output stripped to bare facts.")

    # 构建最终输出
    output = {
        "reasoning_trace": reasoning_trace,
        "derived_metrics": extended_lab_data.get("derived_metrics", {}),
        "clinical_conclusion": current_draft["conclusion"],
        "risk_level_assessment": current_draft["risk_level"]
    }
    return output

# ========== 主函数 ==========
if __name__ == "__main__":
    # 加载示例输入
    with open("sample_inputs/extended_lab_data.json", "r", encoding="utf-8") as f:
        extended_data = json.load(f)
    with open("sample_inputs/patient_context.json", "r", encoding="utf-8") as f:
        patient_ctx = json.load(f)

    # 执行反思循环
    result = run_progressive_pruning(extended_data, patient_ctx)

    # 保存输出
    with open("reasoning_chain.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("System 2 Reflection Loop Completed!")
    print(json.dumps(result, indent=2, ensure_ascii=False))