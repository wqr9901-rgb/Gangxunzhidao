# qc_engine.py  -- 纯代码标注质检规则引擎（确定性、可复现、输出数值证据）
# 支持格式：Labelme JSON / COCO JSON / YOLO txt
# 判定项：框越界、框过小、面积占比、空标注（疑似漏标）
import json
import os
from typing import Dict, Any, List

MIN_BOX_SIZE = 20          # 最小框边长阈值（像素）
MIN_BOX_AREA = 400         # 最小框面积阈值（像素²）
MAX_BOX_AREA_RATIO = 0.95  # 单框面积占整图比例上限（疑似异常大框）
MIN_BOX_AREA_RATIO = 0.0005  # 单框面积占整图比例下限（疑似小目标）

def parse_labelme(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    img_w = int(data.get("imageWidth", 0))
    img_h = int(data.get("imageHeight", 0))
    boxes = []
    for s in data.get("shapes", []):
        label = s.get("label", "unknown")
        pts = s.get("points", [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        boxes.append({
            "label": label,
            "x": x1, "y": y1,
            "w": x2 - x1, "h": y2 - y1
        })
    return boxes, img_w, img_h

def parse_coco(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    img = (data.get("images") or [{}])[0]
    img_w = int(img.get("width", 0))
    img_h = int(img.get("height", 0))
    cat_map = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    boxes = []
    for a in data.get("annotations", []):
        bbox = a.get("bbox", [0, 0, 0, 0])
        boxes.append({
            "label": cat_map.get(a.get("category_id", ""), "unknown"),
            "x": float(bbox[0]), "y": float(bbox[1]),
            "w": float(bbox[2]), "h": float(bbox[3])
        })
    return boxes, img_w, img_h

def parse_yolo(text: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    boxes = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:5])
        boxes.append({
            "label": f"class_{cls}",
            "x": (cx - w / 2) * img_w,
            "y": (cy - h / 2) * img_h,
            "w": w * img_w,
            "h": h * img_h
        })
    return boxes, img_w, img_h

def load_annotation(filename: str, content: bytes) -> Dict[str, Any]:
    """按扩展名解析标注文件，返回标准 boxes 结构。"""
    name = filename.lower()
    if name.endswith(".json"):
        try:
            data = json.loads(content.decode("utf-8"))
        except Exception:
            return {"ok": False, "error": "JSON 解析失败，请检查文件格式。"}
        if "shapes" in data:
            boxes, w, h = parse_labelme(data)
            fmt = "Labelme"
        elif "annotations" in data:
            boxes, w, h = parse_coco(data)
            fmt = "COCO"
        else:
            return {"ok": False, "error": "无法识别 JSON 结构（需为 Labelme 或 COCO 标注格式）。"}
    elif name.endswith(".txt"):
        return {"ok": False, "error": "YOLO txt 需要配合图片尺寸，请同时上传对应图片。"}
    else:
        return {"ok": False, "error": "仅支持 .json（Labelme/COCO）标注文件。"}
    return {"ok": True, "format": fmt, "boxes": boxes, "img_w": w, "img_h": h}

def load_annotation_with_img(filename: str, content: bytes, img_w: int, img_h: int) -> Dict[str, Any]:
    """YOLO txt 需要图片尺寸，走这个入口。"""
    if not filename.lower().endswith(".txt"):
        return load_annotation(filename, content)
    try:
        text = content.decode("utf-8")
    except Exception:
        return {"ok": False, "error": "txt 解析失败。"}
    boxes, w, h = parse_yolo(text, img_w, img_h)
    return {"ok": True, "format": "YOLO", "boxes": boxes, "img_w": w, "img_h": h}

def check_boxes(boxes: List[Dict[str, Any]], img_w: int, img_h: int, image_area: float = 0) -> Dict[str, Any]:
    """确定性规则判定，返回结构化检查结果（含数值证据）。"""
    if img_w <= 0 or img_h <= 0:
        return {"ok": False, "error": "缺少有效的图片尺寸信息，无法校验。"}

    checks = {
        "框是否越界": {"status": "pass", "detail": "", "evidence": []},
        "框是否过小": {"status": "pass", "detail": "", "evidence": []},
        "框面积占比": {"status": "pass", "detail": "", "evidence": []},
        "是否存在明显漏标": {"status": "pass", "detail": "", "evidence": []},
    }
    img_area = image_area if image_area > 0 else img_w * img_h

    if not boxes:
        checks["是否存在明显漏标"] = {
            "status": "warn",
            "detail": "图片存在但没有检出任何标注框，可能整图漏标，请人工复核。",
            "evidence": ["框数量 = 0"]
        }
        return {"ok": True, "box_count": 0, "checks": checks, "summary": "未检出标注框"}

    for i, b in enumerate(boxes):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        label = b.get("label", "unknown")
        tag = f"[框{i+1}·{label}]"

        # 1) 越界
        eps = 1.0
        overflow = []
        if x < -eps:
            overflow.append(f"左边界 x={x:.1f}<0")
        if y < -eps:
            overflow.append(f"上边界 y={y:.1f}<0")
        if x + w > img_w + eps:
            overflow.append(f"右边界 x+w={x+w:.1f}>图宽{img_w}")
        if y + h > img_h + eps:
            overflow.append(f"下边界 y+h={y+h:.1f}>图高{img_h}")
        if overflow:
            checks["框是否越界"]["status"] = "fail"
            checks["框是否越界"]["evidence"].append(f"{tag} 越界：{'；'.join(overflow)}")

        # 2) 过小
        small_reasons = []
        if w < MIN_BOX_SIZE:
            small_reasons.append(f"宽{w:.1f}px<{MIN_BOX_SIZE}px")
        if h < MIN_BOX_SIZE:
            small_reasons.append(f"高{h:.1f}px<{MIN_BOX_SIZE}px")
        if w * h < MIN_BOX_AREA:
            small_reasons.append(f"面积{w*h:.0f}px²<{MIN_BOX_AREA}px²")
        if small_reasons:
            checks["框是否过小"]["status"] = "fail"
            checks["框是否过小"]["evidence"].append(f"{tag} 过小：{'；'.join(small_reasons)}")

        # 3) 面积占比
        ratio = (w * h) / img_area if img_area > 0 else 0
        if ratio > MAX_BOX_AREA_RATIO:
            checks["框面积占比"]["status"] = "warn"
            checks["框面积占比"]["evidence"].append(f"{tag} 面积占比{ratio*100:.1f}%过高，疑似异常大框")
        elif ratio < MIN_BOX_AREA_RATIO:
            checks["框面积占比"]["status"] = "warn"
            checks["框面积占比"]["evidence"].append(f"{tag} 面积占比{ratio*100:.3f}%过低，疑似小目标，请核对阈值规则")

    # 4) 疑似漏标：整图框数过少且图片较大时提示（供人工复核，不做硬判）
    total_box_area = sum(b["w"] * b["h"] for b in boxes)
    coverage = total_box_area / img_area if img_area > 0 else 0
    if len(boxes) == 1 and coverage < 0.05:
        checks["是否存在明显漏标"] = {
            "status": "warn",
            "detail": "整图仅 1 个框且覆盖率极低，可能存在明显漏标，建议人工复核。",
            "evidence": [f"框数量={len(boxes)}，覆盖率={coverage*100:.2f}%"]
        }

    # 汇总
    summary_parts = []
    for k, v in checks.items():
        if v["status"] == "fail":
            summary_parts.append(f"{k} {len(v['evidence'])} 处不通过")
        elif v["status"] == "warn":
            summary_parts.append(f"{k} 有提示")
    summary = "全部通过" if not summary_parts else "；".join(summary_parts)

    return {"ok": True, "box_count": len(boxes), "checks": checks, "summary": summary}
