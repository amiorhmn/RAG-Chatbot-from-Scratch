import requests
from sklearn.metrics.pairwise import cosine_similarity



def get_embeddings(texts: list[str]):
    url = 'http://localhost:11434/api/embed'
    data = {
        "model" : "all-minilm:latest",
        "input" : texts
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=url, json=data, headers=headers).json()
    
    return response['embeddings']



def retrieve(query: str, doc: list[str], doc_embedding):
    query_embedding = get_embeddings([query])
    similarities = cosine_similarity(query_embedding, doc_embedding)
    indexed_similarities = list(enumerate(similarities[0]))
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)

    similar_texts = []
    for i, score in indexed_similarities:
        if score > 0.6:
            similar_texts.append(doc[i])

    return similar_texts



def generate_response(query: str, retrieved_texts: list[str]):
    
    if retrieved_texts == []:
        return "Sorry! No relevant content found in the document"
    
    url = 'http://localhost:11434/api/chat'
    system_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the user question. Give explanation and relevant information using the following context only. Don't make up any new information:
    {' '.join(retrieved_texts)}
    '''
    data = {
        "model" : "gemma3:1b",
        "messages": [
            {
            "role": "system",
            "content": system_prompt
            },
            {
            "role": "user",
            "content": query
            }
        ],
        "stream": False
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=url, json=data, headers=headers).json()
    
    return response['message']['content']