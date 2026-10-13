import bentoml
from legal_rag.api.app import app

@bentoml.service(name="egyptian_civil_code_rag")
@bentoml.asgi_app(app, path="/")
class RAGService:
    pass