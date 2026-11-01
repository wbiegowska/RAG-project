import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY is missing! Please set it in your .env file.")
    st.stop()

from rag.pipeline import run_pipeline  # noqa: E402

st.set_page_config(
    page_title="Warhammer 40k RAG Assistant",
    page_icon="⚔️",
    layout="wide",
)

st.title("⚔️ Warhammer 40k Rules & Units Assistant")
st.write("Ask questions about the Core Rules or unit statistics from the database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "result" in message:
            res = message["result"]
            with st.expander("🔍 View Sources & Details"):
                st.write(f"**Route Chosen:** `{res.route}`")
                st.write(f"**Standalone Query:** `{res.standalone_question}`")

                if res.sql_result and not res.sql_result.error:
                    st.markdown("### 📊 SQL Result")
                    st.code(res.sql_result.sql, language="sql")
                    if res.sql_result.rows:
                        st.dataframe(res.sql_result.rows)
                    else:
                        st.info("Query executed successfully, but returned no rows.")

                if res.retrieved_chunks:
                    st.markdown("### 📖 Retrieved Rule Chunks")
                    for i, chunk in enumerate(res.retrieved_chunks, 1):
                        st.markdown(f"**Chunk {i}**: {chunk}")

if prompt := st.chat_input("Ask a question about Warhammer 40k..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_for_pipeline = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            try:
                pipeline_result = run_pipeline(prompt, history=history_for_pipeline)
                answer = pipeline_result.answer

                st.markdown(answer)

                with st.expander("🔍 View Sources & Details"):
                    st.write(f"**Route Chosen:** `{pipeline_result.route}`")
                    st.write(f"**Standalone Query:** `{pipeline_result.standalone_question}`")

                    if pipeline_result.sql_result and not pipeline_result.sql_result.error:
                        st.markdown("### 📊 SQL Result")
                        st.code(pipeline_result.sql_result.sql, language="sql")
                        if pipeline_result.sql_result.rows:
                            st.dataframe(pipeline_result.sql_result.rows)
                        else:
                            st.info("Query executed successfully, but returned no rows.")

                    if pipeline_result.retrieved_chunks:
                        st.markdown("### 📖 Retrieved Rule Chunks")
                        for i, chunk in enumerate(pipeline_result.retrieved_chunks, 1):
                            st.markdown(f"**Chunk {i}**: {chunk}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "result": pipeline_result}
                )

            except Exception as e:
                st.error(f"An error occurred while running the pipeline: {e}")
