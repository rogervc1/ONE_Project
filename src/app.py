import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from src import config
from src.document_loader import MultiFormatDocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import CorporateRAGChain


st.set_page_config(
    page_title="Base de Conocimiento Corporativa",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%231a1a2e'/><text y='.9em' x='15' font-size='70'>K</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .app-header h1 {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.5;
    }
    .header-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Status indicators */
    .status-bar {
        display: flex;
        gap: 1.2rem;
        margin-top: 1rem;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.active { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.4); }
    .status-dot.warn { background: #f59e0b; }
    .status-dot.off { background: #64748b; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption {
        color: #cbd5e1 !important;
    }
    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-section h4 {
        color: #e2e8f0;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.7rem 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 0.6rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .metric-card .label {
        font-size: 0.65rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.15rem;
    }

    /* Source cards */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .source-card .source-header {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.3rem;
    }
    .source-card .source-meta {
        color: #64748b;
        font-size: 0.72rem;
    }
    .source-card .source-snippet {
        color: #475569;
        font-style: italic;
        margin-top: 0.3rem;
        font-size: 0.78rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 10px !important;
    }

    /* Warning box */
    .config-notice {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.82rem;
        color: #92400e;
        line-height: 1.6;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    loader = MultiFormatDocumentLoader(categories_map=config.CATEGORIES)
    vector_mgr = VectorStoreManager()
    rag_chain = CorporateRAGChain(vector_store_manager=vector_mgr)
    return loader, vector_mgr, rag_chain


def auto_index_if_empty(loader, vector_mgr):
    """Indexa automáticamente los documentos de docs/ si la base está vacía."""
    stats = vector_mgr.get_indexed_stats()
    if stats.get("total_chunks", 0) == 0:
        docs_path = config.DOCS_DIR
        if docs_path.exists() and any(docs_path.rglob("*")):
            all_docs = loader.load_directory(docs_path)
            if all_docs:
                vector_mgr.add_documents(all_docs)
                return True
    return False


def main():
    loader, vector_mgr, rag_chain = load_pipeline()

    # Auto-indexar documentos en el primer inicio
    if "auto_indexed" not in st.session_state:
        auto_index_if_empty(loader, vector_mgr)
        st.session_state.auto_indexed = True

    stats = vector_mgr.get_indexed_stats()
    total_chunks = stats.get("total_chunks", 0)
    categories = stats.get("categories", {})
    has_api_key = bool(os.getenv("GROQ_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY"))

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.01em;">
                Knowledge Base
            </div>
            <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.15rem;">
                Agente Corporativo v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Métricas
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{total_chunks}</div>
                <div class="label">Fragmentos</div>
            </div>
            <div class="metric-card">
                <div class="value">{len(categories)}</div>
                <div class="label">Categorias</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)

        # Filtro por área
        st.markdown('<div class="sidebar-section"><h4>Filtrar por Area</h4></div>', unsafe_allow_html=True)
        categories_options = ["Todas las areas"] + list(config.CATEGORIES.keys())
        selected_category = st.selectbox(
            "Departamento",
            options=categories_options,
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

        # Carga de documentos
        st.markdown('<div class="sidebar-section"><h4>Carga de Documentos</h4></div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Archivos",
            type=["pdf", "csv", "xlsx", "xls", "docx", "pptx", "md", "json", "html", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        target_cat = st.selectbox(
            "Categoria destino",
            options=list(config.CATEGORIES.keys()),
            index=0,
            label_visibility="collapsed"
        )

        if st.button("Procesar e Indexar", use_container_width=True, type="primary"):
            if uploaded_files:
                cat_folder = config.DOCS_DIR / config.CATEGORIES[target_cat]["dir"]
                cat_folder.mkdir(parents=True, exist_ok=True)
                total_added = 0
                with st.spinner("Procesando documentos..."):
                    for uf in uploaded_files:
                        fp = cat_folder / uf.name
                        with open(fp, "wb") as f:
                            f.write(uf.getbuffer())
                        docs = loader.load_document(fp, category_override=target_cat)
                        total_added += vector_mgr.add_documents(docs)
                st.success(f"{len(uploaded_files)} archivo(s) indexado(s) | {total_added} fragmentos")
                st.rerun()
            else:
                st.warning("Selecciona al menos un archivo.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Re-indexar", use_container_width=True):
                with st.spinner("Indexando docs/..."):
                    vector_mgr.clear_vector_store()
                    all_docs = loader.load_directory(config.DOCS_DIR)
                    chunks = vector_mgr.add_documents(all_docs)
                    st.success(f"{chunks} fragmentos indexados")
                    st.rerun()
        with col2:
            if st.button("Limpiar BD", use_container_width=True):
                vector_mgr.clear_vector_store()
                st.info("Base de datos reiniciada.")
                st.rerun()

        # Desglose por categoría
        if categories:
            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section"><h4>Desglose por Area</h4></div>', unsafe_allow_html=True)
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                st.caption(f"{cat}: **{count}** fragmentos")

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        st.caption("Desafio Alura Agentes | Oracle Cloud Infrastructure")

    # ── MAIN AREA ──
    # Header
    dot_class = "active" if has_api_key else "warn"
    api_label = "LLM conectado" if has_api_key else "Sin API Key"
    db_dot = "active" if total_chunks > 0 else "off"
    db_label = f"{total_chunks} fragmentos indexados" if total_chunks > 0 else "Sin documentos"

    st.markdown(f"""
    <div class="app-header">
        <div class="header-badge">Agente de IA Corporativo</div>
        <h1>Base de Conocimiento Interna</h1>
        <p>Consulta politicas, procedimientos y documentos oficiales de la organizacion.<br>
        Las respuestas se fundamentan exclusivamente en los documentos internos indexados.</p>
        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot {dot_class}"></span> {api_label}
            </div>
            <div class="status-item">
                <span class="status-dot {db_dot}"></span> {db_label}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not has_api_key:
        st.markdown("""
        <div class="config-notice">
            <strong>Configuracion requerida:</strong> No se detecto ninguna clave de API en el archivo <code>.env</code>. 
            Agrega <code>GROQ_API_KEY</code>, <code>GOOGLE_API_KEY</code> o <code>OPENAI_API_KEY</code> para habilitar las respuestas conversacionales.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bienvenido a la Base de Conocimiento Corporativa. Puedo responder consultas "
                    "sobre politicas de la empresa fundamentandome en los documentos oficiales indexados.\n\n"
                    "**Algunas consultas que puedes realizar:**\n"
                    "- Cuantos dias de vacaciones me corresponden segun mi antiguedad?\n"
                    "- Cual es el limite de gastos diarios en viajes corporativos?\n"
                    "- Que politica aplica para el uso de herramientas de inteligencia artificial?\n"
                    "- Cual es el SLA de respuesta para incidentes criticos de TI?"
                ),
                "sources": []
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Ver fuentes consultadas"):
                    for src in msg["sources"]:
                        loc_parts = []
                        if src.get("page"): loc_parts.append(f"Pag. {src['page']}")
                        if src.get("row"): loc_parts.append(f"Fila {src['row']}")
                        if src.get("slide"): loc_parts.append(f"Diapositiva {src['slide']}")
                        loc = " | ".join(loc_parts)
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">{src['source']}</div>
                            <div class="source-meta">{src['category']}{(' | ' + loc) if loc else ''}</div>
                            <div class="source-snippet">"{src['snippet']}"</div>
                        </div>
                        """, unsafe_allow_html=True)

    # User input
    user_prompt = st.chat_input("Escribe tu consulta sobre los documentos de la empresa...")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            filter_cat = None if selected_category == "Todas las areas" else selected_category
            with st.spinner("Consultando la base de conocimiento..."):
                response_data = rag_chain.answer_question(user_prompt, category_filter=filter_cat)

            st.markdown(response_data["answer"])

            if response_data.get("sources"):
                with st.expander("Ver fuentes consultadas"):
                    for src in response_data["sources"]:
                        loc_parts = []
                        if src.get("page"): loc_parts.append(f"Pag. {src['page']}")
                        if src.get("row"): loc_parts.append(f"Fila {src['row']}")
                        if src.get("slide"): loc_parts.append(f"Diapositiva {src['slide']}")
                        loc = " | ".join(loc_parts)
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">{src['source']}</div>
                            <div class="source-meta">{src['category']}{(' | ' + loc) if loc else ''}</div>
                            <div class="source-snippet">"{src['snippet']}"</div>
                        </div>
                        """, unsafe_allow_html=True)

            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                st.button("Util", key=f"up_{len(st.session_state.messages)}")
            with col2:
                st.button("Mejorar", key=f"down_{len(st.session_state.messages)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["answer"],
            "sources": response_data.get("sources", [])
        })


if __name__ == "__main__":
    main()
