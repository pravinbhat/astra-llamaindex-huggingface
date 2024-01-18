import os, getpass

api_endpoint = input(
    "\nPlease enter your Database Endpoint URL (e.g. 'https://4bc...datastax.com'):"
)
astra_token = getpass.getpass(
    "\nPlease enter your 'Database Administrator' Token (e.g. 'AstraCS:...'):"
)
os.environ["OPENAI_API_KEY"] = getpass.getpass(
    "\nPlease enter your OpenAI API Key (e.g. 'sk-...'):"
)

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores import AstraDBVectorStore

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
print(f"Total documents: {len(documents)}")
print(f"First document, id: {documents[0].doc_id}")
print(f"First document, hash: {documents[0].hash}")
print(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

astra_db_store = AstraDBVectorStore(
    token=astra_token,
    api_endpoint=api_endpoint,
    collection_name="astra_v_table_uniphore", 
    namespace="vsearch",
    embedding_dimension=1536,
)
storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

print(response.response)

