import json
from typing import Dict, Optional, Any, List

def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def calculate_derived_metrics(
    gender: str,
    age: int,
    lab_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    lab_dict = {}
    for item in lab_items:
        abbr = item["abbreviation"]
        val = safe_float(item["value"])
        lab_dict[abbr] = val

    results = {
        "derived_metrics": {},
        "flags": {},
        "warnings": [],
        "errors": []
    }

    # 1. 阴离子间隙
    na = lab_dict.get("Na+")
    cl = lab_dict.get("Cl-")
    hco3 = lab_dict.get("HCO3-act")
    if na is not None and cl is not None and hco3 is not None:
        ag = na - (cl + hco3)
        results["derived_metrics"]["Anion_Gap"] = round(ag, 2)
        if ag < 10:
            results["flags"]["AG_low"] = True
        elif ag > 20:
            results["flags"]["AG_high"] = True
    else:
        missing = [k for k, v in {"Na+": na, "Cl-": cl, "HCO3-act": hco3}.items() if v is None]
        results["warnings"].append(f"无法计算阴离子间隙：缺少 {', '.join(missing)}")

    # 2. 氧合指数估算
    po2 = lab_dict.get("pO2")
    if po2 is not None:
        fio2_assumed = 0.21
        pao2_fio2 = po2 / fio2_assumed
        results["derived_metrics"]["PaO2_FiO2_estimated"] = round(pao2_fio2, 1)
        results["warnings"].append("氧合指数基于 FiO2=0.21（空气）估算，若吸氧请提供 FiO2")
        if pao2_fio2 < 300:
            results["flags"]["PaO2_FiO2_abnormal"] = True
        if pao2_fio2 < 200:
            results["flags"]["PaO2_FiO2_critical"] = True
    else:
        results["warnings"].append("pO2 缺失，无法估算氧合指数")

    # 3. 酸碱判断
    ph = lab_dict.get("pH")
    pco2 = lab_dict.get("pCO2")
    hco3_act = lab_dict.get("HCO3-act")
    if ph is not None and pco2 is not None and hco3_act is not None:
        acid_base = []
        if ph < 7.35:
            acid_base.append("酸中毒")
            if pco2 > 45:
                acid_base.append("呼吸性")
            if hco3_act < 22:
                acid_base.append("代谢性")
        elif ph > 7.45:
            acid_base.append("碱中毒")
            if pco2 < 35:
                acid_base.append("呼吸性")
            if hco3_act > 27:
                acid_base.append("代谢性")
        else:
            acid_base.append("正常酸碱平衡")
        results["derived_metrics"]["Acid_Base_Assessment"] = " ".join(acid_base)
    else:
        missing = [k for k, v in {"pH": ph, "pCO2": pco2, "HCO3-act": hco3_act}.items() if v is None]
        results["warnings"].append(f"酸碱判断不完整：缺少 {', '.join(missing)}")

    # 4. 乳酸危急值
    lac = lab_dict.get("Lac")
    if lac is not None:
        if lac > 4.0:
            results["flags"]["Lactate_critical"] = True
            results["warnings"].append("乳酸 >4.0 mmol/L，提示组织灌注不足或休克可能")
        elif lac > 2.0:
            results["flags"]["Lactate_elevated"] = True

    # 5. 贫血判断
    hb = lab_dict.get("Hb")
    if hb is not None:
        anemia_threshold = 130 if gender.upper() in ["M", "MALE", "男"] else 120
        if hb < anemia_threshold:
            results["flags"]["Anemia"] = True
            results["derived_metrics"]["Anemia_Threshold_Used_g/L"] = anemia_threshold

    # 6. 钾危急值
    k = lab_dict.get("K+")
    if k is not None:
        if k > 6.0:
            results["flags"]["Hyperkalemia_critical"] = True
        elif k < 2.5:
            results["flags"]["Hypokalemia_critical"] = True

    # 7. 严重低钙
    ca = lab_dict.get("Ca2+")
    if ca is not None and ca < 1.0:
        results["flags"]["Hypocalcemia_severe"] = True

    # 校验 1: tCO2 ≈ HCO3-act + 0.03 * pCO2
    tco2_meas = lab_dict.get("tCO2")
    hco3_act_val = lab_dict.get("HCO3-act")
    pco2_val = lab_dict.get("pCO2")
    
    if tco2_meas is not None and hco3_act_val is not None and pco2_val is not None:
        tco2_calc = hco3_act_val + 0.03 * pco2_val
        diff = abs(tco2_meas - tco2_calc)
        results["derived_metrics"]["tCO2_calculated"] = round(tco2_calc, 2)
        results["derived_metrics"]["tCO2_deviation"] = round(diff, 2)
        
        if diff > 3.0:  # 容差 >3 mmol/L 视为严重不一致
            msg = f"tCO2 实测({tco2_meas}) 与计算值({tco2_calc:.2f}) 偏差过大（Δ={diff:.2f}），OCR 识别可能存在严重误读"
            results["errors"].append(msg)
            results["flags"]["tCO2_OCR_error_suspected"] = True
        elif diff > 2.0:
            results["warnings"].append(f"tCO2 轻微不一致（Δ={diff:.2f}），注意核对")
    else:
        missing = [k for k, v in {"tCO2": tco2_meas, "HCO3-act": hco3_act_val, "pCO2": pco2_val}.items() if v is None]
        results["warnings"].append(f"tCO2 闭环校验跳过：缺少 {', '.join(missing)}")

    # 校验 2: 渗透压 Osm ≈ 2*Na+ + Glu （单位均为 mmol/L）
    osm_meas = lab_dict.get("Osm")
    glu = lab_dict.get("Glu")
    na_val = lab_dict.get("Na+")
    
    if osm_meas is not None and na_val is not None and glu is not None:
        osm_est = 2 * na_val + glu
        diff_osm = abs(osm_meas - osm_est)
        results["derived_metrics"]["Osm_estimated"] = round(osm_est, 1)
        results["derived_metrics"]["Osm_deviation"] = round(diff_osm, 1)
        
        if diff_osm > 20:  # 超过 20 mOsm/kg 视为不匹配
            msg = f"实测渗透压({osm_meas}) 与估算值({osm_est:.1f}) 偏差过大（Δ={diff_osm:.1f}），数据可能不一致"
            results["errors"].append(msg)
            results["flags"]["Osm_inconsistency"] = True
        elif diff_osm > 10:
            results["warnings"].append(f"渗透压轻度不一致（Δ={diff_osm:.1f}）")
    else:
        missing = [k for k, v in {"Osm": osm_meas, "Na+": na_val, "Glu": glu}.items() if v is None]
        results["warnings"].append(f"渗透压校验跳过：缺少 {', '.join(missing)}")

    # ========== 输出整合 ==========
    results["original_lab"] = lab_items
    results["patient_info"] = {"gender": gender, "age": age}

    return results


# ==================== 示例调用（含异常数据测试）====================
if __name__ == "__main__":
    gender = "F"
    age = 55
    lab_data = [
        {"id": 1, "name": "氧合指数", "abbreviation": "PaO2/FiO2", "value": None, "range": "400 ~ 500", "unit": "mmHg"},
        {"id": 2, "name": "酸碱度", "abbreviation": "pH", "value": 7.40, "range": "7.35 ~ 7.45", "unit": ""},
        {"id": 3, "name": "二氧化碳分压", "abbreviation": "pCO2", "value": 40, "range": "35 ~ 45", "unit": "mmHg"},
        {"id": 4, "name": "氧分压", "abbreviation": "pO2", "value": 95, "range": "80 ~ 108", "unit": "mmHg"},
        {"id": 5, "name": "实际碳酸氢盐", "abbreviation": "HCO3-act", "value": 24.0, "range": "22 ~ 27", "unit": "mmol/L"},
        {"id": 6, "name": "二氧化碳总量", "abbreviation": "tCO2", "value": 30.0, "range": "24 ~ 29", "unit": "mmol/L"},  # 故意设高
        {"id": 7, "name": "实际剩余碱", "abbreviation": "BE(act)", "value": 0, "range": "-3 ~ 3", "unit": "mmol/L"},
        {"id": 8, "name": "标准碳酸氢根", "abbreviation": "HCO3-std", "value": None, "range": "22 ~ 27", "unit": "mmol/L"},
        {"id": 9, "name": "标准剩余碱", "abbreviation": "BE(std)", "value": None, "range": "-3 ~ 3", "unit": "mmol/L"},
        {"id": 10, "name": "氧饱和度", "abbreviation": "sO2", "value": 98, "range": "95.0 ~ 99.0", "unit": "%"},
        {"id": 11, "name": "阴离子间隙", "abbreviation": "AG", "value": None, "range": "10 ~ 20", "unit": "mmol/L"},
        {"id": 12, "name": "钾", "abbreviation": "K+", "value": 4.2, "range": "3.5 ~ 4.9", "unit": "mmol/L"},
        {"id": 13, "name": "钠", "abbreviation": "Na+", "value": 140, "range": "135 ~ 145", "unit": "mmol/L"},
        {"id": 14, "name": "氯", "abbreviation": "Cl-", "value": 104, "range": "96 ~ 108", "unit": "mmol/L"},
        {"id": 15, "name": "钙", "abbreviation": "Ca2+", "value": 1.25, "range": "1.15 ~ 1.33", "unit": "mmol/L"},
        {"id": 16, "name": "乳酸", "abbreviation": "Lac", "value": 1.2, "range": "0.5 ~ 1.7", "unit": "mmol/L"},
        {"id": 17, "name": "葡萄糖", "abbreviation": "Glu", "value": 10.0, "range": "3.9 ~ 6.1", "unit": "mmol/L"},
        {"id": 18, "name": "血红蛋白", "abbreviation": "Hb", "value": 118, "range": "110 ~ 150", "unit": "g/L"},
        {"id": 19, "name": "渗透压", "abbreviation": "Osm", "value": 320, "range": "280 ~ 310", "unit": "mmol/kg"},  # 偏高
        {"id": 20, "name": "红细胞压积", "abbreviation": "Hct", "value": 36, "range": "35 ~ 45", "unit": "%"}
    ]

    output = calculate_derived_metrics(gender, age, lab_data)
    print(json.dumps(output, indent=2, ensure_ascii=False))