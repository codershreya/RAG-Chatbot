from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

DB_PATH = 'vectordb'

def load_retriever():
    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return (vector_db, vector_db.as_retriever())