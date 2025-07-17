# AI Chat Assistant with RAG 📚✨

This project implements an AI Chat Assistant utilizing a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on a custom knowledge base. The chatbot features a streaming response interface built with Streamlit [cite: app.py].

## Project Architecture and Flow 🛠️

The project is structured to first preprocess a PDF document, create vectorized embeddings of its content, and then build a RAG pipeline that leverages a local Large Language Model (LLM) for generating responses.

The overall flow is as follows:

1.  **Data Ingestion & Preprocessing**: An input PDF document (`AI Training Document.pdf`) is loaded, its text extracted, cleaned, and then split into smaller, manageable "chunks". These chunks are saved as individual text files in the `chunks/` directory.
2.  **Embedding Creation**: Each text chunk is converted into a numerical vector (embedding) using a pre-trained embedding model. These embeddings are then stored in a `FAISS` vector database (`vectordb/`) for efficient similarity search.
3.  **Retrieval**: When a user poses a question, the `retriever.py` module loads this `FAISS` database and provides a retriever to fetch relevant chunks based on a user's query.
4.  **Generation**: The `generator.py` module defines a prompt template and initializes an LLM. The retrieved chunks, along with the user's question, are passed to this LLM. The LLM then generates a coherent answer based *only* on the provided context.
5.  **RAG Pipeline**: The `pipeline.py` module integrates the retrieval and generation steps using LangChain's Runnable interface. This creates an end-to-end chain that takes a user's question, retrieves relevant context, and generates an answer.
6.  **Chatbot Interface**: The `app.py` file powers a Streamlit web application that serves as the user interface. Users can type questions, and the application interacts with the RAG pipeline. Responses are streamed back to the user, and details about the retrieved chunks are displayed in a sidebar.


## Setup and Installation 🚀

### Prerequisites

Before running the project, ensure you have the following installed:

* **Python 3.8+**
* **Ollama**: This project uses the `mistral:7b-instruct` model, which is run locally via Ollama.
    * Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    * Once Ollama is installed, pull the `mistral:7b-instruct` model by running:
        ```bash
        ollama pull mistral:7b-instruct
        ```

### Project Setup Steps

1.  **Clone the repository (if applicable) or navigate to the project directory.**
2.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    * On Windows: `.\venv\Scripts\activate`
    * On macOS/Linux: `source venv/bin/activate`
4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you have the `requirements.txt` file in your root directory as provided in the previous response.)
5.  **Download NLTK 'punkt' tokenizer data:**
    The `main.ipynb` notebook requires the `punkt` tokenizer for sentence tokenization. You might need to download it if it's not already present. This is done within the notebook.

## Steps to Run Preprocessing, Create Embeddings, and Build the RAG Pipeline 🧠

The `main.ipynb` Jupyter notebook contains all the necessary steps for data preparation, chunking, and creating the vector database. You **must** run this notebook first to set up your knowledge base.

1.  **Place your training document:** Ensure your PDF document is located at `data/AI Training Document.pdf` within your project structure.
2.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Navigate to `notebooks/main.ipynb`.
3.  **Run all cells in `main.ipynb` sequentially:**
    * **Step 1: Loading and Pre-processing the document**: This section reads `data/AI Training Document.pdf`, extracts text, and cleans it.
    * **Step 2: Creating chunks and storing it in a separate directory**: This step tokenizes the cleaned text into sentences, groups them into chunks (of `CHUNK_SIZE = 300` characters), and saves them as `chunk_001.txt`, `chunk_002.txt`, etc., in the `chunks/` directory.
    * **Step 3: Creating Embeddings for our chunks and then storing it in a vector database**: This crucial step takes the created chunks, generates embeddings using `HuggingFaceEmbeddings` with `model_name="all-MiniLM-L6-v2"`, and then builds and saves the `FAISS` vector database into the `vectordb/` directory. This creates `index.faiss` and `index.pkl` files.

    **Note:** The RAG pipeline itself (defined in `pipeline.py`) is "built" dynamically at runtime by importing and using the `retriever` and `generator` modules. The steps above ensure the necessary data (`chunks/` and `vectordb/`) is in place for the pipeline to function.

## Model and Embedding Choices Explained 🧠💡

* **Embedding Model**:
    * **Choice**: `all-MiniLM-L6-v2` from Hugging Face.
    * **Explanation**: This is a powerful and efficient sentence-transformer model widely used for generating dense vector representations of text. These embeddings enable the `FAISS` vector store to quickly find semantically similar text chunks to a given query.

* **Large Language Model (LLM)**:
    * **Choice**: `mistral:7b-instruct`.
    * **Explanation**: This is an instruction-tuned variant of the Mistral 7B model. It is configured to run locally using `OllamaLLM`, allowing for privacy and avoiding reliance on external API services. The temperature is set to `0.3` for more focused and less random responses.

## Instructions to Run the Chatbot with Streaming Response Enabled 💬✨

The chatbot interface is built using Streamlit and inherently supports streaming responses [cite: app.py].

1.  **Ensure the prerequisites are met** (Python packages installed, Ollama running with `mistral:7b-instruct` pulled).
2.  **Make sure you have completed the "Steps to Run Preprocessing, Create Embeddings, and Build the RAG Pipeline"** as the chatbot relies on the `vectordb` and `chunks` directories.
3.  **Navigate to the root directory of your project** (where `app.py` is located).
4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This will open the chat interface in your web browser.

The `app.py` uses `qa_chain.stream(question)` from your `pipeline.py` and `message_placeholder.write_stream()` to deliver the AI's responses word-by-word or token-by-token, providing a dynamic and interactive user experience.

**Example Queries you can try:** 🤔

* "What about fees?"
* "What kind of privacy policies are written here?"
* "How are policies enforced?"
* "Tell me about the legal terms mentioned."


**Demo Video:**
To watch the demo video click [here](https://drive.google.com/file/d/1U7333UmotjEtAfIxC-kQA6trhTPL6IL9/view?usp=sharing)