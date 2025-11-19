# app.py
import os
import re
from datetime import datetime
from datetime import datetime, timedelta
import requests
import streamlit as st
from dotenv import load_dotenv

from rag_qianfan import generate_answer
from trip_storage import (
    create_or_get_trip,
    add_item,
    get_all_trips,
    get_items,
    delete_item,
    update_note,
)

load_dotenv()

# ---------- 工具函数 ----------

def parse_day_slots(day_text: str):
    """
    从某一天的文本中解析出 上午/下午/晚上 的安排
    返回列表：[{time: '上午', text: 'xxx'}, ...]
    """
    slots = []
    pattern = r"-\s*(上午|下午|晚上)[：:]\s*(.+)"
    for m in re.finditer(pattern, day_text):
        time = m.group(1)
        text = m.group(2).strip()
        slots.append({"time": time, "text": text})
    return slots

def extract_places(text: str):
    """
    提取可能是景点/地点的英文短语：
    - 至少两个单词
    - 大写开头
    - 排除 Day / Morning / Afternoon / Evening 等无关词
    """
    # 匹配类似 "Eiffel Tower", "Louvre Museum", "Notre Dame Cathedral"
    pattern = r"\b([A-Z][a-z]+(?:\s+(?:of|the|and|de|la|du|des|[A-Z][a-z]+)){1,3})\b"
    matches = re.findall(pattern, text)

    stopwords = {"Day", "Morning", "Afternoon", "Evening", "注意事项"}
    cleaned = []
    for m in matches:
        head = m.split()[0]
        if head in stopwords:
            continue
        cleaned.append(m.strip())

    return sorted(set(cleaned))


def parse_days(answer: str):
    """
    解析模型输出中的 Day 1 / Day 2 / ... 段落。
    返回: [{'day': 'Day 1 ｜ ...', 'text': '该天对应的全部文本'}, ...]
    """
    pattern = r"(Day\s*\d+[^\n]*)([\s\S]*?)(?=Day\s*\d+|$)"
    matches = re.findall(pattern, answer, flags=re.IGNORECASE)
    blocks = []
    if matches:
        for title, body in matches:
            full = (title + "\n" + body).strip()
            blocks.append({"day": title.strip(), "text": full})
    else:
        # 兜底：如果没匹配到，就把全文当成一个 Day 1
        blocks.append({"day": "Day 1", "text": answer})
    return blocks


def get_weather_summary(city: str):
    """
    统一返回：从今天开始未来 7 天的天气概览
    """
    try:
        today = datetime.today().date()
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=6)).strftime("%Y-%m-%d")

        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        geo_data = geo_resp.json()
        if "results" not in geo_data or len(geo_data["results"]) == 0:
            return "未能找到该城市的天气信息。"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "start_date": start_date,
                "end_date": end_date,
            },
            timeout=5,
        )
        w = weather_resp.json()
        if "daily" not in w:
            return "天气接口暂无数据。"

        daily = w["daily"]
        lines = []
        for date, tmax, tmin, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        ):
            lines.append(f"{date}: 最高 {tmax}°C / 最低 {tmin}°C，降水概率约 {rain}%")

        return "未来 7 天天气概览：\n" + "\n".join(lines)

    except Exception as e:
        return f"获取天气失败：{e}"


# ---------- Streamlit 配置 ----------
st.set_page_config(page_title="AI 旅行助手", layout="wide")
st.title("🌍 AI 旅行助手（基于真实游记 + 千帆大模型）")

# 初始化 session_state
for key, default in [
    ("answer", None),
    ("used_chunks", []),
    ("trip_meta", None),
    ("weather_info", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

tab1, tab2 = st.tabs(["✈️ 规划行程", "⭐ 我的收藏"])

# ---------------- TAB 1：规划行程 ----------------
with tab1:
    st.markdown(
        """
    在这里，你可以描述你想去的城市、出行时间和旅行偏好，我们会：
    1. 先帮你看看目的地的天气情况；
    2. 再从 Reddit / Medium 游记中检索类似行程；
    3. 最后用大模型综合这些信息，给你一份更贴近真实体验的旅行建议。
    """
    )

    with st.sidebar:
        st.header("🧳 你的旅行偏好")
        dest_city = st.text_input("目的地城市（英文/拼音）", value="Paris")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("出发日期", value=datetime.today())
        with col2:
            end_date = st.date_input("结束日期", value=datetime.today())
        trip_style = st.selectbox(
            "旅行风格",
            ["第一次去经典打卡", "小众/本地生活", "亲子友好", "美食为主", "自然风光", "预算友好"],
        )
        pace = st.selectbox("节奏偏好", ["超轻松", "适中", "高强度打卡"])
        companion = st.selectbox(
            "同行人", ["一个人", "情侣/伴侣", "和朋友", "带父母", "带小孩"]
        )
        budget_level = st.selectbox("预算水平", ["穷游", "中等", "偏高", "豪华"])

    st.markdown("### ✏️ 补充说明（可选）")
    user_free_text = st.text_area(
        "可以写下你更具体的期待：例如一定想去哪些地方 / 特别不喜欢什么 / 是否介意走路多：",
        height=120,
    )

    generate_clicked = st.button("生成旅行建议 ✨", key="generate")

    if generate_clicked:
        if not dest_city:
            st.warning("请至少填写目的地城市。")
        else:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            delta_days = (end_date - start_date).days
            days = max(1, delta_days + 1)

            user_question = f"""
目的地：{dest_city}
出行时间：{start_str} ~ {end_str}
旅行风格：{trip_style}
同行人：{companion}
节奏偏好：{pace}
预算水平：{budget_level}

补充说明：{user_free_text or "（用户未补充）"}
"""

            with st.spinner("正在获取天气信息…"):
                weather_info = get_weather_summary(dest_city)

            with st.spinner("正在检索游记并生成建议…"):
                answer, used_chunks = generate_answer(
                    user_question, days=days, top_k=5, city=dest_city
                )

            # 写入 session_state，避免刷新丢失
            st.session_state["answer"] = answer
            st.session_state["used_chunks"] = used_chunks
            st.session_state["trip_meta"] = {
                "city": dest_city,
                "start_date": start_str,
                "end_date": end_str,
                "days": days,
            }
            st.session_state["weather_info"] = weather_info

    # 如果 session_state 里已有结果，就展示
    if st.session_state["answer"]:
        answer = st.session_state["answer"]
        used_chunks = st.session_state["used_chunks"] or []
        trip_meta = st.session_state["trip_meta"]
        weather_info = st.session_state["weather_info"]

        st.markdown("### ☁️ 天气概览")
        st.text(weather_info or "（暂无天气信息）")

        st.markdown("### ✨ 定制旅行建议")
        st.write(answer)

    # 按天解析 & 收藏（改版：按 上午/下午/晚上 收藏整条行程描述）
    st.markdown("### ⭐ 按天收藏行程片段")
    day_blocks = parse_days(answer)
    if not day_blocks:
        st.info("当前回答中没有检测到 Day 结构。")
    else:
        city = trip_meta["city"]
        start_str = trip_meta["start_date"]
        end_str = trip_meta["end_date"]
        trip_id = create_or_get_trip(city, start_str, end_str)

        for block in day_blocks:
            day_label = block["day"]
            day_text = block["text"]

            slots = parse_day_slots(day_text)
            if not slots:
                continue

            st.markdown(f"#### {day_label}")
            for slot in slots:
                time_label = slot["time"]
                text = slot["text"]
                short = text if len(text) <= 40 else text[:40] + "..."
                if st.button(f"收藏：{time_label}｜{short}", key=f"{day_label}_{time_label}_{short}"):
                    # name 存整段描述，day 存 Day1/2/3，time 存 上午/下午/晚上
                    add_item(trip_id, text, day_label, time_label)
                    st.success(f"已收藏 {day_label} {time_label}")

        with st.expander("查看检索到的游记片段（调试用）"):
            for i, r in enumerate(used_chunks):
                md = r.get("metadata", {}) or {}
                url = md.get("url", "")
                title = md.get("title") or md.get("file") or md.get("source", "")
                score = r.get("score", 0.0)
                st.markdown(f"**[{i+1}] {title}** — score: {score:.4f}")
                st.write(r.get("chunk", ""))
                if url:
                    st.write("来源链接:", url)
                st.write("---")

# ---------------- TAB 2：我的收藏 ----------------
with tab2:
    st.header("⭐ 我的收藏行程")
    trips = get_all_trips()

    if not trips:
        st.info("你还没有收藏任何地点，回到“规划行程”生成方案后可以收藏。")
    else:
        for trip in trips:
            trip_id, city, start_date, end_date, title = trip
            st.subheader(f"🗂 {title} — {city}（{start_date} ~ {end_date}）")

            items = get_items(trip_id)
            if not items:
                st.write("（暂无收藏地点）")
            else:
                for item in items:
                    item_id, name, day, time, note = item
                    st.markdown(f"**📍 {name}** — {day or ''} {time or ''}")

                    new_note = st.text_input(
                        f"备注：{name}", value=note or "", key=f"note_{item_id}"
                    )
                    cols = st.columns(2)
                    with cols[0]:
                        if st.button("保存备注", key=f"save_{item_id}"):
                            update_note(item_id, new_note)
                            st.success("已更新备注")
                    with cols[1]:
                        if st.button("删除地点", key=f"delete_{item_id}"):
                            delete_item(item_id)
                            st.warning("已删除该地点")

                st.write("---")