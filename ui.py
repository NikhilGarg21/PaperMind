import os
import streamlit as st

from read_pdf import build_vector_store
from chat import load_chatbot, ask_question

APP_NAME = "PaperMind"
TAGLINE = "AI-Powered Document Question Answering System"

st.set_page_config(
    page_title=f"{APP_NAME} · Document QA",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 1000px;
    }
    .info-box {
        padding: 6px 12px;
        border-radius: 6px;
        border: 1px solid #444;
        margin-bottom: 0.3rem;
    }

    [data-testid="stExpander"] {
        margin-top: 0.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


if "history" not in st.session_state:
    st.session_state.history = []

if "vault_file_names" not in st.session_state:
    st.session_state.vault_file_names = []


def show_sources(chunks):
    if chunks:
        with st.expander(f"Retrieved Chunks ({len(chunks)})"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(
                    f"""
                    **{i}. {chunk['source']}**  
                    Page {chunk['page']}
                    """
                )
                text = chunk["text"]
                st.caption(
                    text[:500] + "..."
                )
                if i != len(chunks):
                    st.markdown(
                        "<hr style='margin:6px 0;'>",
                        unsafe_allow_html=True
                    )


with st.sidebar:
    st.header("📚 PaperMind")
    st.caption(TAGLINE)

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    build_clicked = st.button(
        "Build Vector Store",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


st.title("PaperMind")
st.caption(TAGLINE)

documents = len(st.session_state.vault_file_names)
pages = 0

if "vectorstore" in st.session_state:
    try:
        pages = len(st.session_state.vectorstore.index_to_docstore_id)
    except:
        pages = 0

st.markdown(
    f"""
    <div class="info-box">
        📄 Documents: <b>{documents}</b>
        &nbsp;&nbsp;&nbsp;
        📑 Pages Indexed: <b>{pages}</b>
    </div>
    """,
    unsafe_allow_html=True,
)


if build_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    os.makedirs("data", exist_ok=True)

    for file in os.listdir("data"):
        os.remove(os.path.join("data", file))

    with st.spinner("Building Vector database..."):
        for file in uploaded_files:
            with open(os.path.join("data", file.name), "wb") as f:
                f.write(file.getbuffer())

        build_vector_store()

        vectorstore, chain = load_chatbot()
        st.session_state.vectorstore = vectorstore
        st.session_state.chain = chain
        st.session_state.vault_file_names = [file.name for file in uploaded_files]

    st.success("Knowledge base ready!")
    st.rerun()


st.markdown("### 💬 Chat")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):
        st.write(item["answer"])
        show_sources(item["chunks"])

prompt = st.chat_input("Ask something about your papers...")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    if "chain" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please build the Knowledge Base first.")

    else:
        with st.chat_message("assistant"):
            with st.spinner("Generating Answer..."):
                answer, chunks = ask_question(
                    prompt,
                    st.session_state.vectorstore,
                    st.session_state.chain,
                )

            st.write(answer)
            show_sources(chunks)

        st.session_state.history.append(
            {
                "question": prompt,
                "answer": answer,
                "chunks": chunks,
            }
        )
