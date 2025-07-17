from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .retriever import load_retriever
from .generator import get_llm, get_prompt_template


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": load_retriever()[1] | format_docs,
        "question": RunnablePassthrough()
    } | get_prompt_template() | get_llm() | StrOutputParser()
)