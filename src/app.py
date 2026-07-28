import os
import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from src import config
from src.document_loader import MultiFormatDocumentLoader
from src.vector_store import VectorStoreManager
from src.rag_chain import CorporateRAGChain


# Configuración de Página
st.set_page_config(
    page_title="Agente Corporativo IA | Alura Agentes",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Personalizados para UI Premium
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #2563EB;
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 6px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_pipeline():
    """Inicializa los componentes de VectorDB, Loader y Cadena RAG."""
    loader = MultiFormatDocumentLoader(categories_map=config.CATEGORIES)
    vector_mgr = VectorStoreManager()
    rag_chain = CorporateRAGChain(vector_store_manager=vector_mgr)
    return loader, vector_mgr, rag_chain


def main():
    loader, vector_mgr, rag_chain = load_rag_pipeline()

    # Sidebar: Gestión y Estadísticas
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/robot-corporate.png", width=70)
        st.title("Panel del Agente")
        st.caption("Base de Conocimiento Conversacional")

        st.divider()

        # Filtro de Categoría por Área Corporativa
        st.subheader("🔍 Filtrar por Área")
        categories_options = ["Todas las áreas"] + list(config.CATEGORIES.keys())
        selected_category = st.selectbox(
            "Selecciona un departamento:",
            options=categories_options,
            index=0,
            help="Filtra la búsqueda del agente a los documentos oficiales de un área específica."
        )

        st.divider()

        # Cargar Nuevos Documentos
        st.subheader("📁 Carga de Documentos")
        uploaded_files = st.file_uploader(
            "Subir archivos (PDF, CSV, XLSX, DOCX, PPTX, MD, JSON, HTML):",
            type=["pdf", "csv", "xlsx", "xls", "docx", "pptx", "md", "json", "html"],
            accept_multiple_files=True
        )

        target_cat = st.selectbox(
            "Categoría para los archivos subidos:",
            options=list(config.CATEGORIES.keys()),
            index=0
        )

        if st.button("Procesar e Indexar Documentos", use_container_width=True, type="primary"):
            if uploaded_files:
                cat_folder = config.DOCS_DIR / config.CATEGORIES[target_cat]["dir"]
                cat_folder.mkdir(parents=True, exist_ok=True)
                
                total_added_chunks = 0
                with st.spinner("Procesando e indexando documentos..."):
                    for uploaded_file in uploaded_files:
                        file_path = cat_folder / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Cargar e indexar
                        docs = loader.load_document(file_path, category_override=target_cat)
                        chunks_count = vector_mgr.add_documents(docs)
                        total_added_chunks += chunks_count

                st.success(f"¡{len(uploaded_files)} archivo(s) indexado(s) correctamente ({total_added_chunks} fragmentos)!")
                st.rerun()
            else:
                st.warning("Selecciona al menos un archivo para subir.")

        # Acciones de Mantenimiento
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Indexar docs/", help="Ingesta todos los archivos en la carpeta docs/"):
                with st.spinner("Indexando carpeta docs/..."):
                    all_docs = loader.load_directory(config.DOCS_DIR)
                    chunks = vector_mgr.add_documents(all_docs)
                    st.success(f"¡Se indexaron {chunks} fragmentos!")
                    st.rerun()

        with col_btn2:
            if st.button("Limpiar DB", help="Borra la base de datos vectorial"):
                vector_mgr.clear_vector_store()
                st.info("Base de datos vectorial reiniciada.")
                st.rerun()

        st.divider()

        # Estadísticas de Indexación
        stats = vector_mgr.get_indexed_stats()
        st.metric("Total Chunks Indexados", stats.get("total_chunks", 0))

        if stats.get("categories"):
            with st.expander("Ver Desglose por Área"):
                for cat, count in stats["categories"].items():
                    st.caption(f"• **{cat}**: {count} fragmento(s)")

        st.divider()
        st.caption("Alura Agentes • Despliegue listo para Oracle Cloud Infrastructure (OCI)")

    # Área Principal: Chat Conversacional
    st.markdown('<div class="main-header">🏢 Agente Corporativo de Conocimiento</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Responde a preguntas de los colaboradores basándose en políticas y documentos internos de la empresa.</div>', unsafe_allow_html=True)

    # Estado de la API Key
    has_api_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not has_api_key:
        st.warning(
            "⚠️ **Aviso de Configuración**: No se detectó ninguna `GOOGLE_API_KEY` ni `OPENAI_API_KEY` en el entorno. "
            "El agente funcionará en **modo búsqueda/demostración**, mostrando los extractos recuperados de los documentos. "
            "Para respuestas conversacionales fluidas, agrega tu API Key en el archivo `.env`."
        )

    # Inicializar Historial de Sesión
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Hola! 👋 Soy tu Agente Corporativo de IA. Puedo responder tus dudas sobre políticas de Recursos Humanos, "
                    "procedimientos operativos, temas financieros, contratos legales y más.\n\n"
                    "**Ejemplos de preguntas que puedes hacerme:**\n"
                    "- *¿Cuántos días de vacaciones me corresponden según el tiempo de servicio?*\n"
                    "- *¿Cuál es el procedimiento y límite para el reembolso de gastos de viaje?*\n"
                    "- *¿Qué política de privacidad aplica para el tratamiento de datos de clientes?*"
                ),
                "sources": []
            }
        ]

    # Renderizar Historial de Mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Fuentes de Documentos Consultadas"):
                    for src in msg["sources"]:
                        loc = f"Pág. {src['page']}" if src.get('page') else (f"Fila {src['row']}" if src.get('row') else "")
                        st.markdown(
                            f"<div class='source-box'>"
                            f"📄 <b>{src['source']}</b> ({src['category']}) {loc}<br>"
                            f"<i>\"{src['snippet']}\"</i>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Entrada de Usuario
    user_prompt = st.chat_input("Escribe tu pregunta sobre los documentos de la empresa...")

    if user_prompt:
        # Registrar y mostrar pregunta
        st.session_state.messages.append({"role": "user", "content": user_prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            filter_cat = None if selected_category == "Todas las áreas" else selected_category
            with st.spinner("Consultando la base de conocimiento..."):
                response_data = rag_chain.answer_question(user_prompt, category_filter=filter_cat)

            st.markdown(response_data["answer"])

            if response_data.get("sources"):
                with st.expander("📌 Fuentes de Documentos Consultadas"):
                    for src in response_data["sources"]:
                        loc = f"Pág. {src['page']}" if src.get('page') else (f"Fila {src['row']}" if src.get('row') else "")
                        st.markdown(
                            f"<div class='source-box'>"
                            f"📄 <b>{src['source']}</b> ({src['category']}) {loc}<br>"
                            f"<i>\"{src['snippet']}\"</i>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            # Botones de Feedback
            col_fb1, col_fb2, _ = st.columns([1, 1, 8])
            with col_fb1:
                st.button("👍", key=f"up_{len(st.session_state.messages)}", help="Respuesta útil")
            with col_fb2:
                st.button("👎", key=f"down_{len(st.session_state.messages)}", help="Respuesta incompleta")

        # Guardar en historial
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["answer"],
            "sources": response_data.get("sources", [])
        })


if __name__ == "__main__":
    main()
