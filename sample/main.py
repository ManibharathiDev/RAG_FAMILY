from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag_pipeline import rag_chain, embed_new_text
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader, PyPDFLoader

app = FastAPI()

# CORS for frontend or Spring Boot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat endpoint
class ChatRequest(BaseModel):
    query: str
    user_id: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = rag_chain.invoke(req.query)
    return {"response": response}

# Embed raw text
class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(req: EmbedRequest):
    embed_new_text(req.text)
    return {"status": "success", "message": "Text embedded"}

# Embed uploaded file
@app.post("/embed-file")
async def embed_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    path = f"temp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    if ext == "txt":
        loader = TextLoader(path)
    elif ext == "pdf":
        loader = PyPDFLoader(path)
    else:
        return {"error": "Unsupported file type"}

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vectorstore.add_documents(chunks)

    return {"status": "success", "message": f"{file.filename} embedded"}
