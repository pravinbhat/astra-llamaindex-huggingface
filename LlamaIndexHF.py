import getpass

api_endpoint = input(
    "\nPlease enter your Database Endpoint URL (e.g. 'https://4bc...datastax.com'):"
)
astra_token = getpass.getpass(
    "\nPlease enter your 'Database Administrator' Token (e.g. 'AstraCS:...'):"
)

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
)
from llama_index.llms import HuggingFaceLLM
from llama_index.vector_stores import AstraDBVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

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

llm = HuggingFaceLLM(
    context_window=1024,
    max_new_tokens=256,
    # generate_kwargs={"temperature": 0.25, "do_sample": False},
    tokenizer_name="distilgpt2",
    model_name='distilgpt2',
    device_map="auto",
    tokenizer_kwargs={"max_length": 1024},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)

embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')

service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm, embed_model=embed_model)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context, 
)

query_engine = index.as_query_engine()
response = query_engine.query("Why did the author choose to work on AI?")

print(response.response)
