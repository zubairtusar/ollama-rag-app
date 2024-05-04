import argparse

from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.core import VectorStoreIndex, download_loader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core.embeddings import resolve_embed_model


# Setting argument parser for CLI interactions
parser = argparse.ArgumentParser(
                    prog='Local Bot',
                    description='It replies based on the provided json')
parser.add_argument('-q', '--query', type=str, required=True)
args = parser.parse_args()

print(args.query)

#Load JSON data
JSONReader = download_loader("JSONReader")
loader = JSONReader()
documents = loader.load_data(Path('movies.json'))

#Create Qdrant client and store
client = qdrant_client.QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=client, collection_name="movies")
storage_context = StorageContext.from_defaults(vector_store=vector_store)


Settings.llm = Ollama(model="llama2-uncensored")
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# Create VectorStoreIndex and query engine
index = VectorStoreIndex.from_documents(
                                            documents,
                                            # service_context=service_context,
                                            storage_context=storage_context
                                        )
query_engine = index.as_query_engine()

# Perform a query and print the response
response = query_engine.query(args.query)
print(response)