import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from src import config
from src.vector_store import VectorStoreManager


SYSTEM_PROMPT = """Eres el Agente Corporativo de Inteligencia Artificial de la empresa, una base de conocimiento conversacional abierta para todos los colaboradores.

Tu misión es responder preguntas de manera precisa, profesional y servicial basándote ÚNICAMENTE en el siguiente contexto extraído de nuestros documentos oficiales.

--- CONTEXTO DE DOCUMENTOS INTERNOS ---
{context}
----------------------------------------

REGLAS DE RESPUESTA:
1. Responde directamente a la pregunta usando SOLO la información presente en el contexto anterior.
2. Si el contexto NO contiene la respuesta o la información es insuficiente, responde de forma educada:
   "No he encontrado esta información en los documentos oficiales indexados de la empresa." 
   e indica el contacto del área correspondiente.
3. NO inventes políticas, cifras, fechas ni nombres que no figuren en los documentos.
4. Al final de tu respuesta, incluye siempre una sección titulada "**Fuentes Consultadas:**" listando los documentos exactos utilizados (Nombre de archivo, Categoría y Página/Fila si aplica).

Pregunta del Colaborador: {question}
Respuesta:"""


def get_llm_providers():
    """
    Carga las variables de entorno actualizadas y retorna una lista ordenada de LLMs disponibles.
    Prioriza: Groq (Llama 3.3 70B), Gemini (gemini-2.0-flash-lite / gemini-2.0-flash), OpenAI (gpt-4o-mini).
    """
    load_dotenv(override=True)
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    providers = []

    def is_valid_key(key: str) -> bool:
        return bool(key and not key.startswith("tu_") and not key.startswith("your_") and len(key.strip()) > 15)

    # 1. Groq (Alta velocidad y alta cuota gratuita)
    if is_valid_key(groq_key) and ChatGroq is not None:
        try:
            providers.append(("Groq (Llama 3.3 70B)", ChatGroq(
                model_name="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
                temperature=0.2
            )))
        except Exception as e:
            print(f"[WARN] Error inicializando ChatGroq: {e}")

    # 2. Google Gemini (gemini-2.0-flash-lite y gemini-2.0-flash)
    if is_valid_key(google_key) and ChatGoogleGenerativeAI is not None:
        for m_name in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
            try:
                providers.append((f"Google Gemini ({m_name})", ChatGoogleGenerativeAI(
                    model=m_name,
                    google_api_key=google_key,
                    temperature=0.2
                )))
            except Exception as e:
                print(f"[WARN] Error inicializando ChatGoogleGenerativeAI ({m_name}): {e}")

    # 3. OpenAI
    if is_valid_key(openai_key) and ChatOpenAI is not None:
        try:
            providers.append(("OpenAI (gpt-4o-mini)", ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=openai_key,
                temperature=0.2
            )))
        except Exception as e:
            print(f"[WARN] Error inicializando ChatOpenAI: {e}")

    return providers


class CorporateRAGChain:
    """
    Cadena RAG Corporativa construida con LangChain LCEL.
    Combina búsqueda semántica, filtrado de metadatos, prompts anti-alucinación y citación de fuentes.
    """

    def __init__(self, vector_store_manager: VectorStoreManager = None):
        self.vector_mgr = vector_store_manager or VectorStoreManager()
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def _format_context(self, docs: List[Document]) -> str:
        """Formatea los fragmentos de documentos recuperados para incluirlos en el prompt."""
        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Documento")
            category = doc.metadata.get("category", "General")
            location = ""
            if "page" in doc.metadata:
                location = f" (Pág. {doc.metadata['page']})"
            elif "row" in doc.metadata:
                location = f" (Fila {doc.metadata['row']})"
            elif "slide" in doc.metadata:
                location = f" (Diapositiva {doc.metadata['slide']})"

            header = f"[Fuente {i}: {source} | Área: {category}{location}]"
            formatted_chunks.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(formatted_chunks)

    def _get_fallback_message(self, category_filter: str = None) -> str:
        """Genera un mensaje de fallback amigable si no se encuentran documentos o no hay LLM."""
        cat_info = config.CATEGORIES.get(category_filter, config.CATEGORIES["General / Otro"])
        email = cat_info.get("email", "soporte@empresa.com")
        slack = cat_info.get("slack", "#soporte")

        return (
            "No he encontrado información relevante sobre tu consulta en los documentos oficiales indexados.\n\n"
            f" Te sugerimos ponerte en contacto directamente con el área responsable:\n"
            f"- **Área**: {category_filter or 'Atención General'}\n"
            f"- **Correo**: [{email}](mailto:{email})\n"
            f"- **Canal Interno**: `{slack}`"
        )

    def answer_question(self, question: str, category_filter: str = None) -> Dict[str, Any]:
        """
        Procesa la consulta del usuario mediante el pipeline RAG.
        Intenta proveedores LLM en secuencia y maneja fallbacks.
        """
        # Recargar variables de entorno activamente
        load_dotenv(override=True)

        # 1. Recuperación de documentos semánticamente similares
        retrieved_docs = self.vector_mgr.search_similarity(
            query=question,
            k=4,
            category_filter=category_filter
        )

        if not retrieved_docs:
            return {
                "answer": self._get_fallback_message(category_filter),
                "sources": [],
                "has_context": False
            }

        # 2. Obtener lista de proveedores LLM configurados
        providers = get_llm_providers()

        if not providers:
            context_str = "\n\n".join([f"- **{d.metadata.get('source')}**: {d.page_content[:200]}..." for d in retrieved_docs])
            return {
                "answer": (
                    "⚠️ **Sin API Key Configurada**\n\n"
                    "Se encontraron los siguientes fragmentos relevantes en los documentos de la empresa:\n\n"
                    f"{context_str}\n\n"
                    "*Por favor configura `GOOGLE_API_KEY` o `GROQ_API_KEY` en tu archivo `.env`.*"
                ),
                "sources": [d.metadata for d in retrieved_docs],
                "has_context": True
            }

        # 3. Formatear contexto e intentar la ejecución con los LLMs disponibles
        context_text = self._format_context(retrieved_docs)
        raw_answer = None
        last_error = None

        for name, llm in providers:
            try:
                chain = self.prompt | llm | StrOutputParser()
                raw_answer = chain.invoke({
                    "context": context_text,
                    "question": question
                })
                if raw_answer:
                    break
            except Exception as e:
                last_error = str(e)
                print(f"[WARN] Error al consultar {name}: {last_error[:150]}")
                continue

        # 4. Manejo de respuesta final o error de cuota si todos fallan
        if not raw_answer:
            if last_error and ("429" in last_error or "RESOURCE_EXHAUSTED" in last_error):
                raw_answer = (
                    "🚨 **Cuota Diaria de la API Key Excedida en Google (Limit: 0)**\n\n"
                    "Tu clave de API de Google Gemini en el archivo `.env` pertenece a un proyecto de Google Cloud con la cuota diaria en 0.\n\n"
                    "**Solución en 1 minuto:**\n"
                    "1. Ingresa a **[Google AI Studio](https://aistudio.google.com/app/apikey)**.\n"
                    "2. Haz clic en **Create API key** y selecciona **'Create API key in NEW project'**.\n"
                    "3. Copia esa nueva clave, pégala en tu archivo `.env` (`GOOGLE_API_KEY=...`) y guarda el archivo.\n\n"
                    "*Alternativa:* También puedes usar **Groq API** obteniendo una clave gratis en [console.groq.com/keys](https://console.groq.com/keys) e ingresándola como `GROQ_API_KEY`."
                )
            else:
                raw_answer = f"No se pudo generar la respuesta con la IA. Detalle del error: {last_error or 'Error desconocido'}"

        # 5. Construcción de metadatos de fuentes para la UI
        sources_list = []
        for d in retrieved_docs:
            sources_list.append({
                "source": d.metadata.get("source", "Documento"),
                "category": d.metadata.get("category", "General"),
                "page": d.metadata.get("page"),
                "row": d.metadata.get("row"),
                "slide": d.metadata.get("slide"),
                "snippet": d.page_content[:300] + "..." if len(d.page_content) > 300 else d.page_content
            })

        return {
            "answer": raw_answer,
            "sources": sources_list,
            "has_context": True
        }
