# app.py
import streamlit as st
from dotenv import load_dotenv
import os
from rag_qianfan import generate_answer
from rag_retrieval import search

load_dotenv()  # 从 .env 加载 QIANFAN_API_KEY

st.set_page_config(page_title="AI 旅行助手 RAG", layout="wide")
st.title("🌍 AI 旅行助手（RAG + 千帆）")

with st.sidebar:
    st.markdown("## 设置")
    top_k = st.slider("检索片段数 (top_k)", 1, 10, 5)
    model = st.text_input("千帆模型名", value="ernie-speed-8k")
    temp = st.slider("temperature（创意度）", 0.0, 1.0, 0.2)

st.markdown("在下面输入你的旅行问题（例如：`巴黎三天怎么安排？适合带小孩吗？`）")
query = st.text_input("你的问题", value="")

if st.button("生成旅行建议"):
    if not query.strip():
        st.warning("请输入问题后再生成。")
    else:
        with st.spinner("检索中…"):
            retrieved = search(query, top_k=top_k)
        st.markdown("### 🔎 检索到的片段（用于调试 / 可见来源）")
        for i, r in enumerate(retrieved):
            md = r.get("metadata", {})
            url = md.get("url", "")
            title = md.get("title") or md.get("file") or md.get("source", "")
            st.markdown(f"**[{i+1}] {title}** — score: {r['score']:.4f}")
            st.write(r.get("chunk", ""))
            if url:
                st.write("来源链接:", url)
            st.write("---")

        with st.spinner("生成回答中…"):
            answer, used = generate_answer(query, top_k=top_k, model=model, temperature=temp)

        st.markdown("### ✨ AI 建议（基于检索内容）")
        st.write(answer)

        st.markdown("### 📚 引用的检索片段（编号对应上方）")
        for i, r in enumerate(used):
            md = r.get("metadata", {})
            url = md.get("url", "")
            title = md.get("title") or md.get("file") or md.get("source", "")
            st.markdown(f"- [{i+1}] {title} — {url}")