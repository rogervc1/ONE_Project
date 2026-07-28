import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env o .env.local
load_dotenv()

# Directorios Base
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Asegurar directorios
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Categorías Oficiales de la Empresa y sus Contactos de Fallback
CATEGORIES = {
    "Recursos Humanos": {"dir": "rh", "email": "rh@empresa.com", "slack": "#rh-soporte"},
    "Financiero y Contable": {"dir": "financiero", "email": "finanzas@empresa.com", "slack": "#finanzas-ayuda"},
    "Operacional": {"dir": "operaciones", "email": "operaciones@empresa.com", "slack": "#ops-central"},
    "Estratégico": {"dir": "estrategico", "email": "estrategia@empresa.com", "slack": "#estrategia"},
    "Legal y Compliance": {"dir": "legal", "email": "legal@empresa.com", "slack": "#legal-consulta"},
    "Marketing y Comercial": {"dir": "comercial", "email": "ventas@empresa.com", "slack": "#comercial"},
    "Datos y Sistemas": {"dir": "sistemas", "email": "soporte.it@empresa.com", "slack": "#it-helpdesk"},
    "Investigación y Desarrollo": {"dir": "id", "email": "id@empresa.com", "slack": "#rnd"},
    "Calidad": {"dir": "calidad", "email": "calidad@empresa.com", "slack": "#calidad"},
    "Comunicación Interna": {"dir": "comunicacion", "email": "comunicaciones@empresa.com", "slack": "#anuncios-internos"},
    "General / Otro": {"dir": "general", "email": "contacto@empresa.com", "slack": "#soporte-general"}
}

# Modelo de Embeddings (Por defecto usa CPU HuggingFace liviano si no hay key de OpenAI/Gemini)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo de LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
