from utils import get_embeddings, retrieve, generate_response

file_path = input('Please enter the path of your document: ')
try:
    with open(file_path, encoding='utf-8') as file:
        doc = file.readlines()
        print(f'Document loaded with {len(doc)} entries.')
except Exception:
    raise

doc_embedding = get_embeddings(doc)
try:
    while True:
        input_query = input('Ask a question: ')
        if input_query in ['quit', 'exit']:
            break
        retrieved_knowledge = retrieve(input_query, doc, doc_embedding)
        output_response = generate_response(input_query, retrieved_knowledge)
        print('Chatbot: ' + output_response)
except KeyboardInterrupt:
    print(f'\nExiting due to KeyboardInterrupt.')

if __name__ == '__main__':
    pass