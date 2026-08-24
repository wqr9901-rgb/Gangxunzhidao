# knowledge_base.py
import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

KB_PATH = "data/kb.json"
LEARNED_PATH = "data/learned_notes.json"

@dataclass
class KBItem:
    id: str
    question: str
    answer: str
    tags: List[str]
    category: str
    difficulty: str = "基础"
    source: str = ""

def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _safe_load(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_kb() -> List[Dict[str, Any]]:
    base = _safe_load(KB_PATH, [])
    learned = _safe_load(LEARNED_PATH, [])
    return base + learned

def search_kb(query: str, top_k: int = 5, threshold: float = 0.30) -> List[Dict[str, Any]]:
    kb = load_kb()
    scored = []
    for item in kb:
        q = item.get("question", "")
        a = item.get("answer", "")
        tags = " ".join(item.get("tags", []))
        cat = item.get("category", "")
        text = f"{q}\n{a}\n{tags}\n{cat}"
        q_sim = _sim(query, q)
        full_sim = _sim(query, text)
        score = max(0.65 * q_sim + 0.35 * full_sim, q_sim)
        for kw in item.get("tags", []):
            if kw and kw in query:
                score += 0.08
        if score >= threshold:
            scored.append({**item, "score": round(score, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def kb_direct_answer(query: str, strong_threshold: float = 0.56) -> Tuple[str, List[Dict[str, Any]]]:
    hits = search_kb(query, top_k=5, threshold=0.30)
    if not hits:
        return "", []
    best = hits[0]
    if best["score"] >= strong_threshold:
        return best.get("answer", ""), hits
    return "", hits

def append_learned_qa(question: str, answer: str, tags=None, category="学习沉淀"):
    tags = tags or ["学习沉淀"]
    os.makedirs("data", exist_ok=True)
    learned = _safe_load(LEARNED_PATH, [])
    learned.append({
        "id": f"learned_{len(learned)+1}",
        "question": question,
        "answer": answer,
        "tags": tags,
        "category": category,
        "difficulty": "实践",
        "source": "用户反馈-有帮助"
    })
    with open(LEARNED_PATH, "w", encoding="utf-8") as f:
        json.dump(learned, f, ensure_ascii=False, indent=2)

def classify_question(text: str) -> str:
    t = text or ""
    if any(k in t for k in ["什么是", "定义", "概念", "含义", "是什么意思"]):
        return "定义"
    if any(k in t for k in ["怎么做", "如何", "步骤", "流程", "操作"]):
        return "工具操作"
    if any(k in t for k in ["质检", "抽检", "一致性", "kappa", "漏标", "错标", "重标", "越界", "过小"]):
        return "质检答疑"
    if any(k in t for k in ["规则", "冲突", "规范", "标准"]):
        return "规则咨询"
    if any(k in t for k in ["教案", "课程", "实训安排"]):
        return "教案生成"
    return "综合问答"

def suggested_questions_by_mode(mode: str, weak_dims: List[str]) -> List[str]:
    base = [
        "什么是漏标、错标、重标？怎么区分？",
        "边界框质检有哪些硬性标准？",
        "如何提升标注一致性？",
        "规则冲突时如何判定优先级？",
        "如何写一个可执行的标注规则条款？",
        "请给我一个新手可用的图片标注自检清单。"
    ]
    if "质检答疑" in weak_dims:
        base = ["如何做抽检与复检闭环？", "一致性低于阈值怎么排查？"] + base
    if mode == "专家模式":
        base = ["如何设计多标员共识机制并量化改进收益？"] + base
    return base[:6]




