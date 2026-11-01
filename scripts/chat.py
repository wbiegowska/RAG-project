import pandas as pd
import streamlit as st

from rag.base.pipeline import PipelineResult
from rag.pipeline import run_pipeline
from rag.settings import settings

st.set_page_config(
    page_title="Warhammer 40k RAG Assistant",
    page_icon="⚔️",
    layout="wide",
)

# settings reads the .env of the project, wherever streamlit is started from
if not settings.google_api_key:
    st.error("⚠️ GOOGLE_API_KEY is missing! Please set it in your .env file.")
    st.stop()


def show_details(result: PipelineResult) -> None:
    """Route, rewritten question, SQL and rulebook sources of one answer."""
    with st.expander("🔍 View Sources & Details"):
        st.write(f"**Route Chosen:** `{result.route}`")
        st.write(f"**Standalone Query:** `{result.standalone_question}`")

        sql = result.sql_result
        if sql and sql.error:
            st.warning(f"The SQL query failed: {sql.error}")
            st.code(sql.sql, language="sql")
        elif sql:
            st.markdown("### 📊 SQL Result")
            st.code(sql.sql, language="sql")
            if sql.rows:
                # two columns can have the same name (d.name, m.name), the table needs unique ones
                names = [
                    f"{c} ({i})" if sql.columns.count(c) > 1 else c
                    for i, c in enumerate(sql.columns, 1)
                ]
                st.dataframe(pd.DataFrame(sql.rows, columns=names), hide_index=True)
            else:
                st.info("Query executed successfully, but returned no rows.")

        if result.rule_chunks:
            st.markdown("### 📖 Retrieved Rule Chunks")
            for i, chunk in enumerate(result.rule_chunks, 1):
                st.markdown(f"**{i}. {chunk.section_path}** (page {chunk.page})")
                st.markdown(chunk.text)


st.title("⚔️ Warhammer 40k Rules & Units Assistant")
st.write("Ask questions about the Core Rules or unit statistics from the database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message:
            show_details(message["result"])

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
            except Exception as e:
                st.error(f"An error occurred while running the pipeline: {e}")
                st.stop()
        st.markdown(pipeline_result.answer)
        show_details(pipeline_result)

    st.session_state.messages.append(
        {"role": "assistant", "content": pipeline_result.answer, "result": pipeline_result}
    )
