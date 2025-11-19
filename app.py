# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from rag_qianfan import generate_answer
from rag_retrieval import search
import requests
from trip_storage import create_or_get_trip, add_item, get_all_trips, get_items, delete_item, update_note 
import re
def extract_places(text):
    pattern = r"\b([A-Z][a-zA-Z\s'-]{2,})\b"
    matches = re.findall(pattern, text)
    return list(set(matches))


# --------- 天气工具函数（和上面给的一样） ----------
def get_weather_summary(city: str, start_date: str, end_date: str):
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5
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
            timeout=5
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
            daily["precipitation_probability_max"]
        ):
            lines.append(f"{date}: 最高 {tmax}°C / 最低 {tmin}°C，降水概率约 {rain}%")

        return "未来天气概览：\n" + "\n".join(lines[:3])

    except Exception as e:
        return f"获取天气失败：{e}"


# --------- Streamlit 界面 ----------
st.set_page_config(page_title="AI 旅行助手", layout="wide")
st.title("🌍 AI 旅行助手（基于真实游记 + 千帆大模型）")

tab1, tab2 = st.tabs(["✈️ 规划行程", "⭐ 我的收藏"])

# ---------------- TAB 1：规划行程 ----------------
with tab1:
    st.markdown("""
    在这里，你可以描述你想去的城市、出行时间和旅行偏好，我们会：
    1. 先帮你看看目的地的天气情况；
    2. 再从 Reddit / Medium 游记中检索类似行程；
    3. 最后用大模型综合这些信息，给你一份更贴近真实体验的旅行建议。
    """)

    with st.sidebar:
        st.header("🧳 你的旅行偏好")
        dest_city = st.text_input("目的地城市（英文/拼音）", value="Paris")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("出发日期", value=datetime.today())
        with col2:
            end_date = st.date_input("结束日期", value=datetime.today())
        trip_style = st.selectbox("旅行风格", ["第一次去经典打卡", "小众/本地生活", "亲子友好", "美食为主", "自然风光", "预算友好"])
        pace = st.selectbox("节奏偏好", ["超轻松", "适中", "高强度打卡"])
        companion = st.selectbox("同行人", ["一个人", "情侣/伴侣", "和朋友", "带父母", "带小孩"])
        budget_level = st.selectbox("预算水平", ["穷游", "中等", "偏高", "豪华"])

    st.markdown("### ✏️ 补充说明（可选）")
    user_free_text = st.text_area("可以写下你更具体的期待：例如一定想去哪些地方 / 特别不喜欢什么 / 是否介意走路多：", height=120)

    if st.button("生成旅行建议 ✨", key="generate"):
        if not dest_city:
            st.warning("请至少填写目的地城市。")
        else:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            user_question = f'''
            目的地：{dest_city}
            出行时间：{start_str} ~ {end_str}
            旅行风格：{trip_style}
            同行人：{companion}
            节奏偏好：{pace}
            预算水平：{budget_level}

            补充说明：{user_free_text or "（用户未补充）"}

            请基于这些条件，为我设计一份合适的旅行建议。
            '''

            with st.spinner("正在获取天气信息…"):
                weather_info = get_weather_summary(dest_city, start_str, end_str)

            st.markdown("### ☁️ 天气概览")
            st.text(weather_info)

            with st.spinner("正在检索游记并生成建议…"):
                answer, used_chunks = generate_answer(user_question, top_k=5, model="ernie-speed-8k")
                # 保存结果到 session_state，避免重新运行时丢失
                st.session_state["last_answer"] = answer
                st.session_state["last_used_chunks"] = used_chunks
                st.session_state["last_trip_id"] = (dest_city, start_str, end_str)

            st.markdown("### ✨ 定制旅行建议")
            st.write(answer)

            st.markdown("### ⭐ 可收藏的地点")
            answer = st.session_state.get("last_answer", answer)
            places = extract_places(answer)
            city, start_str, end_str = st.session_state["last_trip_id"]
            real_trip_id = create_or_get_trip(city, start_str, end_str)

            for p in places:
                if st.button(f"收藏：{p}", key=f"save_{p}"):
                    real_trip_id = create_or_get_trip(dest_city, start_str, end_str)
                    add_item(real_trip_id, p, "", "")
                    st.success(f"已收藏 {p}")

            with st.expander("查看检索到的游记片段（调试用）"):
                for i, r in enumerate(used_chunks):
                    md = r.get("metadata", {})
                    url = md.get("url", "")
                    title = md.get("title") or md.get("file") or md.get("source", "")
                    st.markdown(f"**[{i+1}] {title}** — score: {r['score']:.4f}")
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

                    new_note = st.text_input(f"备注：{name}", value=note, key=f"note_{item_id}")
                    if st.button(f"保存备注：{name}", key=f"save_{item_id}"):
                        update_note(item_id, new_note)
                        st.success("已更新备注")

                    if st.button(f"删除：{name}", key=f"delete_{item_id}"):
                        delete_item(item_id)
                        st.warning("已删除该地点")

                st.write("---")