from aibot import AIBot
from torch import bfloat16, float16
import sys
import os
from extractor import text_from_pdf
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import glob

# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
# )

# Open the DB
db_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        # embedding_func=embedding_func,
        persist_directory="./database"
    )
)

# list all collections
db_client.list_collections()

# get an existing collection
try:
    collection = db_client.get_collection("Documents")
except ValueError:
    # if not there, create the collection
    collection = db_client.create_collection(name="Documents")
    for file_name in glob.glob("documents/*.pdf"):
        pdf_text = text_from_pdf(file_name)
        collection.add(
            documents=pdf_text['text'],
            metadatas=pdf_text['meta'],
            ids=["ID-"+str(id) for id in range(len(pdf_text['text']))]
        )


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = 'stabilityai/stablelm-zephyr-3b'

chat = AIBot()
chat.initialize(model_name,
                max_length=4096,
                load_in_8bit=True,
                load_in_4bit=False,
                trust_remote_code=True,
                temperature=0.2,
                top_k=10,
                top_p=2,
                repetition_penalty=2.0,
                do_sample=True,
                torch_dtype=float16, # torch.bfloat16
                device_map='cuda')

while 1:
    system_prompt = """
    You are an friendly and useful assistant. Use this information to answer any question as simple as possible. You will only answer about these areas:\n
    """

    user_query = input('Ask anything: ')
    print()
    
    results = collection.query(
        query_texts=[user_query],
        n_results=2
    )

    for result in results['documents'][0]:
        # print(result)
        system_prompt += result + "\n"
        
    text_prompt = f"<|system|>{system_prompt}\s\n<|user|>{user_query}\s<|assistant|>"
    previous_index = len(text_prompt)
    
    ln = 1
    print()
    print("---------------")
    accumulator = ""
    while ln > 0:
        chunk = chat.generate_text(text_prompt,
                                 max_new_tokens=2,
                                 repetition_penalty=2.,
                                 stops_when_find_token=True)
        ln = len(chunk)
        accumulator += chunk
        # print(accumulator)
        if chat.text_has_end(accumulator):
            break
        else:
            text_prompt += chunk
            output = text_prompt[previous_index:]
            sys.stdout.write('\r'+output)
            sys.stdout.flush()
    print()
    print("---------------")
    print(text_prompt[previous_index:])
    print()
    print("---------------")



