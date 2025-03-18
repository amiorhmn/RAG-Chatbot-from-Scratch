from utils import get_embeddings, retrieve, generate_response

file_path = input('Please enter the path of your document: ')
try:
    with open(file_path, encoding='utf-8') as file:
        doc = file.readlines() # doc is a list of strings where every element is a line from the text file
        print(f'Document loaded with {len(doc)} entries.')
except Exception:
    raise # raise exception if no file is found in the path or the file is of invalid format

doc_embedding = get_embeddings(doc) # the embedding matrix
try:
    while True:
        input_query = input('Ask a question: ')
        if input_query in ['quit', 'exit']: # keywords to trigger exit from the application
            break
        retrieved_knowledge = retrieve(input_query, doc, doc_embedding) # list of relevant texts from the document
        output_response = generate_response(input_query, retrieved_knowledge) # the answer string of the LLM response
        print('Chatbot: ' + output_response)
except KeyboardInterrupt:
    print(f'\nExiting due to KeyboardInterrupt.') # exit if pressed ctrl+c

if __name__ == '__main__':
    pass