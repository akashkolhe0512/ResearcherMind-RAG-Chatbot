import os
import streamlit as st

from ingest import load_documents, text_splitter, embed_and_store
from retriever import load_vectorstore, build_retriever
from chain import rag_chain, ask

st.set_page_config(
    page_title="ResearcherMind",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f1117;
        color: #ffffff;
    }

    /* Answer box styling */
    .answer-box {
        background-color: #1e2130;
        border-left: 4px solid #7c3aed;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.7;
    }

    /* Chunk preview box */
    .chunk-box {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 6px;
        padding: 12px;
        margin: 6px 0;
        font-size: 13px;
        color: #a0aec0;
    }

    /* Header styling */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 16px;
        margin-top: 4px;
    }

    /* Upload area */
    .upload-section {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 24px;
        border: 1px dashed #374151;
    }
</style>
""", unsafe_allow_html=True)

# Session state persists values across reruns of the app
# (Streamlit reruns the whole script on every interaction)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None       # holds the built RAG chain

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # list of (question, answer) tuples

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False  # tracks if PDF is ready

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None     # name of current PDF


# Header
st.markdown('<p class="main-title">🧠 ResearchMind</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a research paper. Ask anything. Get grounded answers with citations.</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### 📄 Upload Research Paper")
    st.markdown("Drop a PDF and click **Process** to index it.")

    uploaded_file = st.file_uploader(
        label="Choose a PDF",
        type=["pdf"],
        help="Upload any research paper in PDF format"
    )

    if uploaded_file:
        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📎 **{uploaded_file.name}**\n\n{file_size_mb:.1f} MB")

        process_btn = st.button(
            "⚡ Process PDF",
            type="primary",
            use_container_width=True
        )

        if process_btn:
            # Only reprocess if it's a new PDF
            if uploaded_file.name != st.session_state.current_pdf:

                with st.spinner("📖 Reading PDF..."):
                    # Save uploaded file to data/ folder
                    save_path = os.path.join("data", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                with st.spinner("✂️ Splitting into chunks..."):
                    documents = load_documents("data")
                    chunks = text_splitter(documents)

                with st.spinner("🔢 Embedding chunks (this takes ~30 seconds)..."):
                    vectorstore = embed_and_store(chunks, "vectorstore")

                with st.spinner("🔗 Building RAG chain..."):
                    retriever = build_retriever(vectorstore)
                    st.session_state.rag_chain = rag_chain(retriever)
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf = uploaded_file.name
                    st.session_state.chat_history = []  # reset chat for new PDF

                st.success("✅ Ready! Ask your questions.")

            else:
                st.info("This PDF is already processed!")

    # Sidebar stats
    if st.session_state.pdf_processed:
        st.divider()
        st.markdown("### 📊 Session Info")
        st.metric("PDF loaded", st.session_state.current_pdf[:20] + "...")
        st.metric("Questions asked", len(st.session_state.chat_history))

    st.divider()
    st.markdown("### ⚙️ About")
    st.markdown("""
    **Stack:**
    - 🦜 LangChain
    - 🟣 ChromaDB
    - 🤗 HuggingFace Embeddings
    - ⚡ Groq (Llama 3)
    - 🎈 Streamlit
    """)

# Two column layout
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### 💬 Ask a Question")

    # Question input
    question = st.text_input(
        label="Your question",
        placeholder="e.g. What is the main contribution of this paper?",
        label_visibility="collapsed"
    )

    # Suggested questions
    st.markdown("**Suggested questions:**")
    suggestion_cols = st.columns(2)

    suggestions = [
        "What is the main contribution?",
        "What datasets were used?",
        "What are the limitations?",
        "How does the proposed method work?",
        "What were the evaluation results?",
        "Who are the authors?"
    ]

    # When user clicks a suggestion, it fills the question
    for i, suggestion in enumerate(suggestions):
        col = suggestion_cols[i % 2]
        if col.button(suggestion, key=f"sug_{i}", use_container_width=True):
            question = suggestion

    st.divider()

    # Ask button
    ask_btn = st.button(
        "🔍 Ask ResearchMind",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.pdf_processed
    )

    # Handle question submission
    if ask_btn and question:
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first.")
        else:
            with st.spinner("🧠 Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)

            # Save to chat history
            st.session_state.chat_history.append((question, answer))

    # If PDF not uploaded yet, show placeholder
    if not st.session_state.pdf_processed:
        st.info("👈 Upload a PDF from the sidebar to get started.")


with col2:
    st.markdown("### 📚 Retrieved Chunks")
    st.markdown("*The source passages used to generate the answer*")

    if st.session_state.pdf_processed and question and ask_btn:
        # Show the chunks that were retrieved for the last question
        vectorstore = load_vectorstore()
        retriever = build_retriever(vectorstore)
        chunks = retriever.invoke(question)

        for i, chunk in enumerate(chunks):
            page = chunk.metadata.get("page", "?")
            with st.expander(f"📄 Chunk {i+1} — Page {page}", expanded=(i==0)):
                st.markdown(f'<div class="chunk-box">{chunk.page_content}</div>',
                           unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chunk-box">
        Retrieved chunks will appear here after you ask a question.
        They show exactly which parts of the paper the AI used
        to generate its answer — making the system fully transparent.
        </div>
        """, unsafe_allow_html=True)


# Show previous Q&A pairs
if st.session_state.chat_history:
    st.divider()
    st.markdown("### 🕘 Conversation History")

    # Show most recent first
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        turn = len(st.session_state.chat_history) - i

        with st.expander(f"Q{turn}: {q[:60]}...", expanded=(i == 0)):
            st.markdown("**Question:**")
            st.info(q)
            st.markdown("**Answer:**")
            st.markdown(f'<div class="answer-box">{a}</div>',
                       unsafe_allow_html=True)

    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()