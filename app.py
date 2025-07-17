import streamlit as st

from src.retriever import load_retriever
from src.pipeline import qa_chain

import os


st.set_page_config(page_title="AI Chat Interface", page_icon="✨")
st.title("AI Chat Assistant ✨ ")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "sidebar_data" not in st.session_state:
    st.session_state.sidebar_data = {
        "model": "N/A",
        "chunks": []
    }


def clear_chat_history():
    st.session_state.messages = [] 
    st.session_state.sidebar_data = {
        "model": "N/A",
        "chunks": []
    }
    st.rerun()


def load_chunks(vector_db, retriever, question):
    chunk_ids = [doc.id for doc in retriever.invoke(question)]
    reversed_chunks_dict = {value: key for key, value in vector_db.index_to_docstore_id.items()}
    chunks = [int(reversed_chunks_dict[ids]) for ids in chunk_ids]

    CHUNK_DIRECTORY = "chunks"
    current_chunks_data = []

    for chunk_id in chunks:
        chunk_filename = os.path.join(CHUNK_DIRECTORY, f"chunk_{chunk_id+1:03d}.txt")

        with open(chunk_filename, 'r', encoding='utf-8') as file:
            content = file.read()

        current_chunks_data.append({
            "chunk_id": chunk_id+1,
            "content": content,
            "length": len(content),
        })
    
    st.session_state.sidebar_data["chunks"] = current_chunks_data

def get_response(question):
    vector_db, _retriever = load_retriever()
    load_chunks(vector_db, _retriever, question)
    return qa_chain.stream(question)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        with st.status("AI is thinking...", expanded=True) as status:
            message_placeholder = st.empty()

            status.update(label="Generating response...", state="running", expanded=True)

            full_response = message_placeholder.write_stream(get_response(prompt))

            status.update(label="Response received!", state="complete", expanded=True)


    st.session_state.sidebar_data["model"] = f"mistral:7b-instruct"
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    with st.sidebar:
        st.header("Model Information")
        st.markdown(f"**Model Used:** `{st.session_state.sidebar_data['model']}`")

        st.subheader("Chunk Details")
        if st.session_state.sidebar_data["chunks"]:
            for i, chunk in enumerate(st.session_state.sidebar_data["chunks"]):
                st.markdown(f"**Chunk {chunk['chunk_id']}:**")
                st.code(f"Content: \"{chunk['content'][:30]}...\"", language='text', wrap_lines=True)
                st.markdown(f"Length: `{chunk['length']} characters`")
                st.markdown("---")
        else:
            st.info("No chunk data available yet. Start a conversation!")


st.button("Clear Chat", on_click=clear_chat_history)