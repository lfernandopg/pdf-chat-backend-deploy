## 📚 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/v1/chats/upload` | Crear chat con PDFs |
| `GET` | `/api/v1/chats/{chat_id}/status` | Verificar estado del chat |
| `POST` | `/api/v1/chats/{chat_id}/add-documents` | Añadir PDFs al chat |
| `POST` | `/api/v1/chats/{chat_id}/ask` | Hacer pregunta |
| `DELETE` | `/api/v1/chats/{chat_id}` | Eliminar chat |

## 🎮 Uso

### 1. Crear un Chat
```python
import requests

files = {'files': open('documento.pdf', 'rb')}
response = requests.post(
    'http://localhost:8000/api/v1/chats/upload',
    files=files
)
chat_id = response.json()['chat_id']
```

### 2. Hacer Pregunta
```python
response = requests.post(
    f'http://localhost:8000/api/v1/chats/{chat_id}/ask',
    json={
        'prompt': '¿Cuál es el tema principal?',
        'llm_model_name': 'gpt-3.5-turbo'
    }
)
answer = response.json()['answer']


## 📁 Estructura del Proyecto

```
.
├── backend/                # Backend FastAPI
│   ├── src/
│   │   ├── api/           # Endpoints REST
│   │   ├── services/      # Lógica de negocio
│   │   ├── repositories/  # Capa de datos
│   │   ├── models/        # Modelos Pydantic
│   │   └── core/          # Configuración
│   ├── main.py
│   └── requirements.txt
│
```