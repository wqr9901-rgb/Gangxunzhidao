
import os
import json
import base64
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

import streamlit as st
from openai import OpenAI
from PIL import Image

from secret import SPARK_API_PASSWORD
from knowledge_base import (
    kb_direct_answer, search_kb, append_learned_qa,
    classify_question, suggested_questions_by_mode
)
import qc_engine

# ========= 基本配置 =========
st.set_page_config(page_title="AI数据标注工程师助教", page_icon="🎓", layout="wide")
os.makedirs("data", exist_ok=True)

MODEL_NAME = "4.0Ultra"
BASE_URL = "https://spark-api-open.xf-yun.com/v1/"

client = OpenAI(api_key=SPARK_API_PASSWORD, base_url=BASE_URL)

HISTORY_PATH = "data/chat_history.json"
FEEDBACK_PATH = "data/feedback.json"
PROFILE_PATH = "data/learner_profile.json"
GROWTH_PATH = "data/growth.json"

DIMENSIONS = ["规则咨询", "工具操作", "质检答疑", "定义理解", "教案执行"]

# 回答卡片浅灰底色 + 图片校验卡片样式
CARD_CSS = """
<style>
.stChatMessage.stChatMessageAssistant > [data-testid="stChatMessageContent"] {
    background-color: #F4F4F5;
    border-radius: 12px;
    padding: 14px 18px;
    border: 1px solid #E4E4E7;
}
.qc-card {
    border: 1px solid #E4E4E7;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    background: #FFFFFF;
}
.qc-card.pass { border-left: 4px solid #16A34A; }
.qc-card.warn { border-left: 4px solid #F59E0B; }
.qc-card.fail { border-left: 4px solid #DC2626; }
.qc-tag { font-weight: 700; font-size: 13px; }
.qc-tag.pass { color: #16A34A; }
.qc-tag.warn { color: #F59E0B; }
.qc-tag.fail { color: #DC2626; }
.qc-detail { color: #52525B; font-size: 13px; margin-top: 4px; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ========= 工具函数 =========
def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = load_json(HISTORY_PATH, [])
    if "mode" not in st.session_state:
        st.session_state.mode = "新手模式"
    if "module" not in st.session_state:
        st.session_state.module = "规则咨询"
    if "last_qa" not in st.session_state:
        st.session_state.last_qa = {"q": "", "a": "", "label": "综合问答"}
    if "profile" not in st.session_state:
        st.session_state.profile = load_json(PROFILE_PATH, {k: 0 for k in DIMENSIONS})
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = ""

def persist():
    save_json(HISTORY_PATH, st.session_state.messages)
    save_json(PROFILE_PATH, st.session_state.profile)

def log_growth(dim: str, delta: int):
    """记录一次能力变化（时间 + 维度 + 增量 + 当前分），用于成长曲线。"""
    growth = load_json(GROWTH_PATH, [])
    score = st.session_state.profile.get(dim, 0)
    growth.append({
        "time": datetime.now().isoformat(timespec="minutes"),
        "dim": dim,
        "delta": delta,
        "score": score
    })
    save_json(GROWTH_PATH, growth[-1000:])

def map_label_to_dim(label: str):
    if label == "定义":
        return "定义理解"
    if label == "教案生成":
        return "教案执行"
    if label in DIMENSIONS:
        return label
    if label == "综合问答":
        m = st.session_state.module
        if m in DIMENSIONS:
            return m
    return None

def skill_level(score: int) -> str:
    if score >= 80:
        return "精通"
    if score >= 50:
        return "熟练"
    if score >= 20:
        return "入门"
    return "待提升"

def build_system_prompt(mode: str, module: str):
    p = f"""
你是“AI数据标注工程师助教”。
要求：
1) 回答详细、步骤化、可执行；
2) 每次回答末尾必须给一个反问；
3) 如果是定义问题，严格结构：
   - 专业定义
   - 通俗解释
   - 示例
   - 常见误区
   - 反问引导
4) 回答末尾必须用“===思考过程===”块给出教学版思考过程，包含四个小节：
   - 思路摘要（一句话概括整体思路）
   - 解题步骤（分步说明怎么处理这个问题）
   - 判断依据（依据哪些知识、规则、检索结果）
   - 为什么这样答（站在教学角度解释这样引导的目的）
   思考块以“===END===”结束，内容要贴合本次回答，不要照抄模板；
5) 思考过程用于给学生看“怎么想到的”，要正面、简短，禁止展示内部逐字推理；
6) 如果不确定，明确写出假设与边界；
7) 优先遵循本地知识库和项目规则，避免泛化跑偏。
当前模式：{mode}
当前模块：{module}
"""
    if mode == "新手模式":
        p += "\n语言尽量通俗、每步解释原因。"
    else:
        p += "\n可以使用专业术语，并补充指标与流程优化建议。"
    return p

def stream_llm(messages: List[Dict[str, Any]]):
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        stream=True
    )
    for chunk in stream:
        try:
            c = chunk.choices[0].delta.content
            if c:
                yield c
        except Exception:
            continue

def fallback_think(label: str) -> str:
    return f"""💡 思路摘要：问题被识别为「{label}」类型，采用对应答题框架。

📋 解题步骤：
1. 先检索本地知识库，判断是否存在标准口径；
2. 未命中高置信答案时，结合项目规则与上下文组织回答；
3. 按「{label}」的专属结构输出，末尾给出反问引导。

⚖️ 判断依据：基于问题关键词的维度分类与知识库检索置信度。

🎯 为什么这样答：教学视角上，既给结论又教方法，帮助新手建立“先检索、再判断、后表达”的解题习惯。"""

def split_think(full: str) -> Tuple[str, str]:
    start = full.find("===思考过程===")
    if start == -1:
        return full, ""
    end_marker = "===END==="
    end = full.find(end_marker)
    if end == -1:
        end = len(full)
    think = full[start + len("===思考过程==="):end].strip()
    tail = full[end + len(end_marker):].strip()
    body = full[:start].rstrip()
    if tail:
        body = body + "\n" + tail
    return body, think

def answer_with_rag(user_q: str):
    direct, hits = kb_direct_answer(user_q)
    kb_ctx = "\n\n".join([f"[{h.get('category','')}] Q:{h.get('question','')}\nA:{h.get('answer','')}" for h in hits[:3]])

    is_def = classify_question(user_q) == "定义"

    if direct:
        answer = f"""### 知识库标准答案
{direct}

### 反问引导
你能用你当前项目里的一个样本，自己判断一次这个概念吗？
"""
        return answer, "kb"

    system = build_system_prompt(st.session_state.mode, st.session_state.module)
    if is_def:
        up = f"""
问题：{user_q}

检索到的参考知识：
{kb_ctx if kb_ctx else "无"}

请严格按以下结构输出：
### 专业定义
### 通俗解释
### 示例
### 常见误区
### 反问引导
最后追加“===思考过程===”教学版思考块（思路摘要/解题步骤/判断依据/为什么这样答），以“===END===”结束。
"""
    else:
        up = f"""
问题：{user_q}

检索到的参考知识：
{kb_ctx if kb_ctx else "无"}

请按以下结构输出：
### 直接回答
### 分步骤操作
### 常见错误与质检要点
### 反问引导
最后追加“===思考过程===”教学版思考块（思路摘要/解题步骤/判断依据/为什么这样答），以“===END===”结束。
"""

    recent = st.session_state.messages[-8:]
    llm_msgs = [{"role": "system", "content": system}] + recent + [{"role": "user", "content": up}]
    return llm_msgs, "llm"

def image_to_base64(file):
    return base64.b64encode(file.read()).decode("utf-8")

def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return None

def render_qc_checks(checks: List[Dict[str, str]]):
    if not checks:
        return
    for c in checks:
        rule = c.get("rule", "")
        status = c.get("status", "warn")
        detail = c.get("detail", "")
        tag = {"pass": "通过", "warn": "提示", "fail": "不通过"}.get(status, "提示")
        color = status if status in ("pass", "warn", "fail") else "warn"
        st.markdown(
            f'<div class="qc-card {color}"><span class="qc-tag {color}">{tag} · {rule}</span>'
            f'<div class="qc-detail">{detail}</div></div>',
            unsafe_allow_html=True
        )

def review_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        w, h = img.size
        basic = f"图片尺寸：{w}x{h}；模式：{img.mode}。"
    except Exception:
        basic = ""

    try:
        uploaded_file.seek(0)
        b64 = image_to_base64(uploaded_file)
        rule_prompt = (
            "你是数据标注质检专家。如果图片是标注截图（有标注框），请按以下规则逐项校验，"
            "并只输出一个 JSON 对象（不要多余文字），格式："
            '{"is_annotation_screenshot": true, "checks": ['
            '{"rule": "框是否越界", "status": "pass|warn|fail", "detail": "说明与建议"},'
            '{"rule": "类别是否缺失", "status": "pass|warn|fail", "detail": "说明与建议"},'
            '{"rule": "框是否过小", "status": "pass|warn|fail", "detail": "说明与建议"},'
            '{"rule": "是否存在明显漏标", "status": "pass|warn|fail", "detail": "说明与建议"}]}'
            "如果图片不是标注截图，则输出 {\"is_annotation_screenshot\": false, \"checks\": []}。"
            "校验要结合可见标注框：框越界看是否超出图像边界；类别缺失看类别标签是否齐全；"
            "框过小看是否有明显小于正常目标尺寸的框；漏标看是否有显著目标未被打框。"
        )
        msgs = [
            {"role": "system", "content": "你是数据标注质检助教，输出 JSON。"},
            {"role": "user", "content": [
                {"type": "text", "text": rule_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ]
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            temperature=0.2,
            max_tokens=1000
        )
        text = resp.choices[0].message.content
        data = extract_json(text)
        if data and data.get("is_annotation_screenshot"):
            checks = data.get("checks", [])
            return {"mode": "qc", "basic": basic, "checks": checks}
        return {"mode": "note", "basic": basic, "text": text}
    except Exception:
        return {"mode": "fallback", "basic": basic}

def generate_lesson_plan(topic: str, level: str = "新手"):
    prompt = f"""
请为“AI数据标注实训”生成可直接执行的教案，主题：{topic}，学员水平：{level}。
输出结构：
1) 课程目标（知识/技能/质量意识）
2) 先修要求
3) 90分钟详细流程（每10-15分钟一个环节）
4) 演示样例与练习题（含标准答案要点）
5) 质检评分rubric（含扣分项）
6) 常见误区与纠偏策略
7) 课后作业与复盘模板
8) 教师提示词（可直接投喂给助教系统）
最后加一个课堂提问。
"""
    msgs = [{"role":"system","content":"你是资深数据标注教学设计专家。"},
            {"role":"user","content":prompt}]
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=msgs, temperature=0.5, max_tokens=1800
    )
    return resp.choices[0].message.content

# ========= 侧边栏 =========
init_state()

with st.sidebar:
    st.title("功能导航")
    st.session_state.mode = st.radio("学习模式", ["新手模式", "专家模式"], index=0)
    st.session_state.module = st.selectbox("学习模块", ["规则咨询", "工具操作", "质检答疑"], index=0)

    # ---- 能力图谱 ----
    st.markdown("### 🧭 能力图谱")
    total = sum(st.session_state.profile.values())
    avg = total // len(DIMENSIONS) if DIMENSIONS else 0
    st.metric("综合能力值", f"{total}", help="各维度能力分总和，点“有用”按问题类型加分")
    st.caption(f"综合等级：{skill_level(avg)}（{avg}/100）")

    import pandas as pd
    chart_data = pd.DataFrame({
        "维度": DIMENSIONS,
        "能力分": [int(st.session_state.profile.get(d, 0)) for d in DIMENSIONS]
    })
    st.vega_lite_chart(
        chart_data,
        {
            "mark": {"type": "bar", "cornerRadius": 6},
            "encoding": {
                "y": {"field": "维度", "type": "nominal", "sort": "-x", "axis": {"labelFontSize": 13}},
                "x": {"field": "能力分", "type": "quantitative", "scale": {"domain": [0, 120]}},
                "color": {"value": "#3B82F6"}
            },
            "height": 220
        },
        width="stretch"
    )

    for d in DIMENSIONS:
        score = int(st.session_state.profile.get(d, 0))
        st.progress(min(score, 100), text=f"{d}：{score} 分 · {skill_level(score)}")

    # ---- 能力成长曲线 ----
    growth = load_json(GROWTH_PATH, [])
    if growth:
        st.markdown("#### 📈 能力成长曲线")
        gdf = pd.DataFrame(growth)
        try:
            gdf["time"] = pd.to_datetime(gdf["time"])
        except Exception:
            pass
        st.vega_lite_chart(
            gdf,
            {
                "mark": {"type": "line", "point": True, "strokeWidth": 2},
                "encoding": {
                    "x": {"field": "time", "type": "temporal", "title": "时间", "axis": {"labelAngle": -40, "labelFontSize": 10}},
                    "y": {"field": "score", "type": "quantitative", "title": "能力分", "scale": {"domain": [0, 120]}},
                    "color": {"field": "dim", "type": "nominal", "title": "维度"}
                },
                "height": 230
            },
            width="stretch"
        )

    weak = sorted(st.session_state.profile.items(), key=lambda x: x[1])[:2]
    weak_dims_list = [w[0] for w in weak if w[1] < 50]
    st.caption("薄弱维度：" + ("、".join(weak_dims_list) if weak_dims_list else "暂无，表现均衡"))
    if weak_dims_list:
        st.info("建议优先学习：" + "、".join(weak_dims_list))

    st.markdown("---")
    if st.button("保存会话"):
        persist()
        st.success("已保存")
    if st.button("清空会话"):
        st.session_state.messages = []
        persist()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📷 图片批改")
    imgs = st.file_uploader("上传标注截图，自动做规则校验", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    st.markdown("---")
    st.markdown("### 🧪 标注文件质检（规则引擎）")
    ann_files = st.file_uploader(
        "上传标注文件（Labelme/COCO JSON，或 YOLO txt）",
        type=["json", "txt"],
        accept_multiple_files=True
    )
    ann_img = st.file_uploader(
        "（YOLO txt 需要）上传对应图片以获得尺寸",
        type=["png", "jpg", "jpeg"],
        key="ann_img"
    )

# ========= 主页面 =========
st.title("🎓 AI数据标注工程师助教")
st.caption("本系统基于国产大模型算力构建，数据本地化处理，保障实训数据安全。")

weak_dims = [w[0] for w in weak]
recs = suggested_questions_by_mode(st.session_state.mode, weak_dims)
st.markdown("### 推荐问题")
cols = st.columns(3)
for i, q in enumerate(recs):
    with cols[i % 3]:
        if st.button(q, key=f"rec_{i}"):
            st.session_state.pending_q = q

# 图片批改展示
if imgs:
    st.markdown("## 📷 图片批改 · 标注规则校验")
    for i, f in enumerate(imgs):
        st.image(f, caption=f"样本 {i+1}", width=240)
        result = review_image(f)
        mode = result.get("mode")
        if mode == "qc":
            st.markdown(f"**规则校验结果**（{result.get('basic','')}）")
            render_qc_checks(result.get("checks", []))
            st.markdown("**反问引导**：这张图里你最不确定的是‘漏标’还是‘边界是否贴合’？")
        elif mode == "note":
            st.markdown(result.get("basic", ""))
            st.markdown(result.get("text", "图片非标注截图，跳过规则校验。"))
            st.markdown("**反问引导**：如果要把它当作标注样本检查，你会重点看哪个规则？")
        else:
            st.markdown("### 图片批改结果（降级模式）")
            st.markdown(result.get("basic", ""))
            st.markdown(
                "当前接口可能未开启视觉能力，先给你通用质检清单：\n"
                "1. 是否存在漏标（显著目标未标）；\n"
                "2. 类别是否错标（相近类别混淆）；\n"
                "3. 边界框是否越界；\n"
                "4. 边界框是否过小/过松/过紧；\n"
                "5. 遮挡目标是否按规则处理；\n"
                "6. 类别字段是否齐全；\n"
                "7. 同类目标标注标准是否一致。\n"
            )
            st.markdown("**反问引导**：如果你自己先复检一遍，你觉得最可能错在哪一项？")

# 标注文件质检展示（纯规则引擎，确定性校验）
if ann_files:
    st.markdown("## 🧪 标注文件质检 · 规则引擎")
    for i, f in enumerate(ann_files):
        st.markdown(f"**文件 {i+1}**：`{f.name}`（{f.size} 字节）")
        content = f.read()
        if f.name.lower().endswith(".txt"):
            if ann_img is None:
                st.warning("YOLO txt 需要图片尺寸，请在左侧上传对应图片后再质检。")
                continue
            from PIL import Image as _Img
            ann_img.seek(0)
            pw, ph = _Img.open(ann_img).size
            result = qc_engine.load_annotation_with_img(f.name, content, pw, ph)
        else:
            result = qc_engine.load_annotation(f.name, content)

        if not result.get("ok"):
            st.error(result.get("error", "解析失败"))
            continue

        st.caption(f"格式：{result['format']} ｜ 框数量：{result.get('box_count', len(result.get('boxes', [])))} ｜ 图片尺寸：{result.get('img_w')}x{result.get('img_h')}")
        check_res = qc_engine.check_boxes(result.get("boxes", []), result.get("img_w", 0), result.get("img_h", 0))
        if not check_res.get("ok"):
            st.error(check_res.get("error", "校验失败"))
            continue

        # 渲染每个检查项（含数值证据）
        for rule, c in check_res["checks"].items():
            status = c["status"]
            detail = c["detail"]
            evidence = c["evidence"]
            ev_txt = "；".join(evidence) if evidence else "未发现问题"
            body = f"{detail} {ev_txt}".strip()
            if status == "pass":
                tag = "通过"
            elif status == "warn":
                tag = "提示"
            else:
                tag = "不通过"
            st.markdown(
                f'<div class="qc-card {status}"><span class="qc-tag {status}">{tag} · {rule}</span>'
                f'<div class="qc-detail">{body}</div></div>',
                unsafe_allow_html=True
            )
        st.markdown(f"**结论**：{check_res['summary']}")
        st.markdown("---")

    st.info("说明：规则引擎为确定性硬校验（越界/过小/占比/漏标线索），结果可复现；漏标与类别等高阶判断仍由多模态大模型补充。")

# 历史消息渲染（含思考折叠 + 反馈按钮）
def render_feedback(msg_key: str, label: str, q: str, a: str):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("👍 有用", key=f"help_{msg_key}"):
            dim = map_label_to_dim(label)
            if dim:
                st.session_state.profile[dim] = st.session_state.profile.get(dim, 0) + 8
                log_growth(dim, +8)
            append_learned_qa(q, a, tags=[label, st.session_state.module], category=label)
            fb = load_json(FEEDBACK_PATH, [])
            fb.append({"time": datetime.now().isoformat(), "q": q, "a": a, "feedback": "helpful", "label": label})
            save_json(FEEDBACK_PATH, fb)
            persist()
            st.toast(f"已记录，{dim or '综合能力'}能力分 +8。")
            st.rerun()
    with c2:
        if st.button("👎 没用", key=f"bad_{msg_key}"):
            dim = map_label_to_dim(label)
            if dim:
                st.session_state.profile[dim] = max(0, st.session_state.profile.get(dim, 0) - 3)
                log_growth(dim, -3)
            fb = load_json(FEEDBACK_PATH, [])
            fb.append({"time": datetime.now().isoformat(), "q": q, "a": a, "feedback": "not_helpful", "label": label})
            save_json(FEEDBACK_PATH, fb)
            persist()
            st.toast(f"已记录，{dim or '综合能力'}能力分 -3，将优先优化这一类问题。")
            st.rerun()

for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            think = m.get("think", "")
            if think:
                with st.expander("🧠 思考过程（思路摘要 / 解题步骤 / 判断依据 / 为什么这样答）", expanded=False):
                    st.markdown(think)
            q = m.get("q", st.session_state.last_qa.get("q", ""))
            label = m.get("label", st.session_state.last_qa.get("label", "综合问答"))
            msg_key = m.get("id") or f"idx_{idx}"
            render_feedback(msg_key, label, q, m["content"])

# ========= 输入处理 =========
user_q = st.chat_input("请输入问题，或输入“生成教案：主题”")

if st.session_state.pending_q:
    user_q = st.session_state.pending_q
    st.session_state.pending_q = ""

if user_q:
    label = classify_question(user_q)

    st.session_state.messages.append({
        "id": f"m{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "role": "user",
        "content": user_q,
        "label": label,
        "q": user_q
    })
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        final_ans = ""
        think_text = ""
        if user_q.startswith("生成教案：") or label == "教案生成":
            topic = user_q.replace("生成教案：", "").strip() or "数据标注质检入门"
            plan = generate_lesson_plan(topic, "新手" if st.session_state.mode == "新手模式" else "进阶")
            final_ans = plan
            think_text = fallback_think("教案执行")
            st.markdown(final_ans)
        else:
            payload, source_type = answer_with_rag(user_q)
            if source_type == "kb":
                final_ans = payload
                think_text = fallback_think(label)
                st.markdown(final_ans)
            else:
                placeholder = st.empty()
                full = ""
                for tok in stream_llm(payload):
                    full += tok
                    placeholder.markdown(full)
                body, think = split_think(full)
                final_ans = body if body else full
                think_text = think or fallback_think(label)
                placeholder.markdown(final_ans)

        with st.expander("🧠 思考过程（思路摘要 / 解题步骤 / 判断依据 / 为什么这样答）", expanded=False):
            st.markdown(think_text)

    st.session_state.messages.append({
        "id": f"m{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "role": "assistant",
        "content": final_ans,
        "think": think_text,
        "label": label,
        "q": user_q
    })
    st.session_state.last_qa = {"q": user_q, "a": final_ans, "label": label}
    persist()
    st.rerun()









  




      