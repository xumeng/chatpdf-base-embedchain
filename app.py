import os
import queue
import re
import tempfile
import threading

import streamlit as st

from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import StreamingStdOutCallbackHandlerYield, generate


def embedchain_bot_by_mistral(db_path):
    return App.from_config(
        config={
            "app": {
                "config": {
                    "collect_metrics": False,
                }
            },
            "llm": {
                "provider": "huggingface",
                "config": {
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "temperature": 0.5,
                    "top_p": 0.5,
                    "stream": True,
                    "max_tokens": 8000,
                },
            },
            "vectordb": {
                "provider": "chroma",
                "config": {
                    "collection_name": "chat-pdf",
                    "dir": db_path,
                    "allow_reset": True,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-mpnet-base-v2"},
            },
            "chunker": {
                "chunk_size": 2000,
                "chunk_overlap": 0,
                "length_function": "len",
            },
        }
    )


def get_db_path():
    tmpdirname = tempfile.mkdtemp()
    return tmpdirname


def get_ec_app():
    if "app" in st.session_state:
        print("Found app in session state")
        app = st.session_state.app
    else:
        print("Creating app")
        db_path = get_db_path()
        app = embedchain_bot_by_mistral(db_path)
        st.session_state.app = app
    return app


def app_response(prompt, result):
    llm_config = app.llm.config.as_dict()
    # llm_config["callbacks"] = [StreamingStdOutCallbackHandlerYield(q=q)]
    config = BaseLlmConfig(**llm_config)
    answer, citations = app.chat(prompt, config=config, citations=True)
    result["answer"] = answer
    result["citations"] = citations


# 获取回复内容
# 此处因embedchain原来通过线程和队列的实现方式测试有问题，改为直接获取结果并通过正则解析
def get_answer(text):
    print(">>>>>>>", text)
    match = re.search(r"(.*)\nAnswer:\n(.+)", text, re.DOTALL)
    if match:
        answer_content = match.group(2)
        return answer_content
    return ""


# summarize the doc
def summarize():
    results = {}
    prompt = "Write a concise summary of the content, No more than 50 words"
    app_response(prompt, results)
    summary = get_answer(results["answer"])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"This is a summary:\n\n {summary}",
        }
    )


with st.sidebar:
    app = get_ec_app()

    pdf_files = st.file_uploader(
        "Upload your PDF files", accept_multiple_files=True, type="pdf"
    )
    add_pdf_files = st.session_state.get("add_pdf_files", [])
    for pdf_file in pdf_files:
        file_name = pdf_file.name
        if file_name in add_pdf_files:
            continue
        try:
            temp_file_name = None
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, prefix=file_name, suffix=".pdf"
            ) as f:
                f.write(pdf_file.getvalue())
                temp_file_name = f.name
            if temp_file_name:
                st.markdown(f"Adding {file_name} to knowledge base...")
                app.add(temp_file_name, data_type="pdf_file")
                st.markdown("")
                add_pdf_files.append(file_name)
                os.remove(temp_file_name)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Added {file_name} to knowledge base!",
                }
            )
            summarize()
        except Exception as e:
            st.error(f"Error adding {file_name} to knowledge base: {e}")
            st.stop()
    st.session_state["add_pdf_files"] = add_pdf_files

st.title("📄 Chat with PDF - by Embedchain")
styled_caption = """
> <b>基于Embedchain实现的PDF对话。<b>
> * LLM model: Mistral-7B-Instruct<br>
> * VecetorDB: Chroma<br>
> * Embedder: sentence-transformers/all-mpnet-base-v2
"""
st.markdown(styled_caption, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hi! I'm chatbot, which can answer questions about your pdf documents.\n
                Upload your pdf documents here and I'll answer your questions about them! 
            """,
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    app = get_ec_app()

    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(prompt)

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("Thinking...")
        full_response = ""

        # q = queue.Queue()
        results = {}
        app_response(prompt, results)
        # thread = threading.Thread(target=app_response, args=(results,))
        # thread.start()

        # for answer_chunk in generate(q):
        #     full_response += answer_chunk
        #     msg_placeholder.markdown(full_response)

        full_response = get_answer(results["answer"])
        msg_placeholder.markdown(full_response)

        # thread.join()
        answer, citations = results["answer"], results["citations"]
        if citations:
            full_response += "\n\n**Sources**:\n"
            sources = []
            for i, citation in enumerate(citations):
                source = citation[1]["url"]
                pattern = re.compile(r"([^/]+)\.[^\.]+\.pdf$")
                match = pattern.search(source)
                if match:
                    source = match.group(1) + ".pdf"
                sources.append(source)
            sources = list(set(sources))
            for source in sources:
                full_response += f"- {source}\n"

        msg_placeholder.markdown(full_response)
        print("Answer: ", full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
