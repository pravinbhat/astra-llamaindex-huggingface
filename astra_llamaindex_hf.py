import os
import getpass

import torch

api_endpoint = 'https://full-db-id.apps.astra.datastax.com'
token = 'AstraCS:token'

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
)
from llama_index.vector_stores import AstraDBVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

astra_db_store = AstraDBVectorStore(
    token=token,
    api_endpoint=api_endpoint,
    collection_name="your-astra-table", 
    namespace="your-astra-namespace",
    embedding_dimension=1536,
)

storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

from transformers import AutoModel

access_token = "hf_token"
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name=model,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

print(response.response)
