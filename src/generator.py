from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
    You are a helpful and honest assistant.
    Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    Context: {context}

    Question: {question}
"""

def get_prompt_template():
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question']
    )
    return prompt

def get_llm():
    llm = OllamaLLM(
        model="mistral:7b-instruct",
        temperature=0.3
    )
    return llm