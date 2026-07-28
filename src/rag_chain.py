import os
from typing import List, Dict, Any

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

Respuesta:"""


def get_llm():
    """
    Inicializa el LLM según las claves disponibles en el entorno.
    Soporta Groq (Llama 3.3 70B), Google Gemini (gemini-2.0-flash) y OpenAI (gpt-4o-mini).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key and ChatGroq is not None:
        try:
            return ChatGroq(
                model_name="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
                temperature=0.2
            )
        except Exception as e:
            print(f"[WARN] Error inicializando ChatGroq: {e}")

    if google_key and ChatGoogleGenerativeAI is not None:
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=google_key,
                temperature=0.2
            )
        except Exception as e:
            print(f"[WARN] Error inicializando ChatGoogleGenerativeAI con gemini-2.0-flash-lite: {e}")

    if openai_key and ChatOpenAI is not None:
        try:
            return ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=openai_key,
                temperature=0.2
            )
        except Exception as e:
            print(f"[WARN] Error inicializando ChatOpenAI: {e}")

    return None


class CorporateRAGChain:
    """
    Cadena RAG Corporativa construida con LangChain LCEL.
    Combina búsqueda semántica, filtrado de metadatos, prompts anti-alucinación y citación de fuentes.
    """

    def __init__(self, vector_store_manager: VectorStoreManager = None):
        self.vector_mgr = vector_store_manager or VectorStoreManager()
        self.llm = get_llm()
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
        Retorna la respuesta generada y la lista estructurada de fuentes utilizadas.
        """
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

        # 2. Si no hay LLM configurado (API Key faltante), retornar extractos recuperados
        if self.llm is None:
            context_str = "\n\n".join([f"- **{d.metadata.get('source')}**: {d.page_content[:200]}..." for d in retrieved_docs])
            return {
                "answer": (
                    "⚠️ **Modo Demostración sin API Key de LLM**\n\n"
                    "Se encontraron los siguientes fragmentos relevantes en los documentos de la empresa:\n\n"
                    f"{context_str}\n\n"
                    "*Nota: Configura `GOOGLE_API_KEY` o `OPENAI_API_KEY` en el archivo `.env` para habilitar la generación fluida de respuestas por IA.*"
                ),
                "sources": [d.metadata for d in retrieved_docs],
                "has_context": True
            }

        # 3. Formateo de contexto y ejecución de la cadena LCEL
        context_text = self._format_context(retrieved_docs)
        raw_answer = None

        google_key = os.getenv("GOOGLE_API_KEY")
        # Probar modelos Gemini válidos (gemini-2.0-flash-lite primero por velocidad y cuota)
        gemini_candidates = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]

        if google_key and ChatGoogleGenerativeAI is not None:
            for model_name in gemini_candidates:
                try:
                    llm_candidate = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=google_key,
                        temperature=0.2
                    )
                    chain = self.prompt | llm_candidate | StrOutputParser()
                    raw_answer = chain.invoke({
                        "context": context_text,
                        "question": question
                    })
                    break  # Respuesta generada con éxito
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        raw_answer = (
                            "⏳ **Límite de Consultas Alcanzado (Rate Limit de Gemini)**\n\n"
                            "La API gratuita de Google Gemini ha alcanzado temporalmente el límite de peticiones por minuto (15 RPM).\n"
                            "Por favor espera **30 a 60 segundos** e intenta tu pregunta nuevamente."
                        )
                        break
                    else:
                        print(f"[WARN] Error al consultar modelo {model_name}: {err_str[:150]}")
                        continue

        if not raw_answer and self.llm is not None:
            try:
                chain = self.prompt | self.llm | StrOutputParser()
                raw_answer = chain.invoke({
                    "context": context_text,
                    "question": question
                })
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    raw_answer = (
                        "⏳ **Límite de Consultas Alcanzado (Rate Limit de Gemini)**\n\n"
                        "La API gratuita de Google Gemini ha alcanzado temporalmente el límite de peticiones por minuto (15 RPM).\n"
                        "Por favor espera **30 a 60 segundos** e intenta tu pregunta nuevamente."
                    )
                else:
                    raw_answer = f"Error al generar la respuesta con la IA: {err_str}"

        if not raw_answer:
            raw_answer = "No se pudo conectar con el servicio de IA. Revisa tu clave de API en el archivo `.env`."


        # 4. Construcción de metadatos de fuentes para la UI
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
