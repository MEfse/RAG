from transformers import pipeline

generator = pipeline('text-generation', model='gpt-2')

def generate_answer(context):
    response = generator(f"Answer the following question: {context}")
    return response[0]['generated_text']