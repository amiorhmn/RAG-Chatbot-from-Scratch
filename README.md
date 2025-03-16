# RAG-Chatbot-from-Scratch
A simple command-line RAG (Retieval Augmented Generation) chatbot application built from scratch. No external framework or library has been used except *requests* and *scikit-learn* modules. It uses a locally hosted LLM via *Ollama* to generate embedding and response to user's query based on relevant information from a text file.

## Features
- No external RAG framework or agent library
- Uses locally hosted open-source models
- Command-line interface
- Handling of error during document loading
- Exit program via `KeyboardInterrupt` or `exit` and `quit` keywords

## Conversation Example
```
(.venv) PS C:\Users\Amio\Desktop\RAG-Chatbot-from-Scratch> python main.py
Please enter the path of your document: cat-facts.txt
Document loaded with 150 entries.
Ask a question: What do cats eat?
Chatbot: According to the context, cats eat about five mice per meal.
Ask a question: How long does a cat live? 
Chatbot: According to the provided context, the average life span of a domestic cat is 14 years.
Ask a question: quit                         
(.venv) PS C:\Users\Amio\Desktop\RAG-Chatbot-from-Scratch>
```

## Requirements
- *Ollama* for hosting open-source embedding and large language models
- *Requests* library for communicating with the models
- *Scikit-learn* library for calculating `cosine-similarity` during retrieval step

This specific example uses *Gemma3* model for response generation and *all-minilm* for embedding generation. The models can be changed from the `utils.py` file.

## Structure of the code
The application contains two files:
- main.py
- utils.py
  - `get_embeddings`
  - `retrieve`
  - `generate_response`

The `utils.py` contains three functions. Each of them does a specific task in the RAG pipeline. The app is run from the `main.py` file.

## How to run
1. Install *Ollama* from [ollama.com](https://ollama.com)
2. Download an embedding model and a text model (e.g. *all-minilm* and *Gemma3*)
   ```
   ollama pull all-minilm
   ollama pull gemma3
   ```
3. Clone the repository
   ```
   git clone https://github.com/amiorhmn/RAG-Chatbot-from-Scratch.git
   cd RAG-Chatbot-from-Scratch
   ```
4. Create and activate a virtual environment
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```
5. Install the requirements
   ```
   pip install -r requirements.txt
   ```
6. Run the app
   ```
   python main.py
   ```
