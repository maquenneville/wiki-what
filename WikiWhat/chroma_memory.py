# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 00:08:42 2023

@author: marca
"""

import chromadb
from embedder import Embedder
import configparser
import os
import time
import openai
from openai.error import RateLimitError, InvalidRequestError, APIError
import tiktoken
from tqdm.auto import tqdm
import sys
import random
from chromadb.utils import embedding_functions
import asyncio
import nest_asyncio

nest_asyncio.apply()

# =============================================================================
# class ChromaMemory:
#     def __init__(self, collection_name=None, storage=None):
#         if not os.path.exists("config.ini"):
#             raise FileNotFoundError("The config file was not found.")
# 
#         self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# 
#         (
#             self.openai_api_key,
#             self.chroma_collection,
#         ) = self._get_api_keys("config.ini")
#         
#         self.ef = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key=self.openai_api_key,
#                 model_name="text-embedding-ada-002"
#             )
#         
#         if collection_name:
#             self.chroma_collection = collection_name.lower()
#             
#         self.client = chromadb.Client()
#         self.storage = storage
#         
#         if self.storage:
#             self.client = chromadb.PersistentClient(path=self.storage)
#             
#         self.collection = self.client.create_collection(self.chroma_collection, embedding_function=self.ef, metadata={"hnsw:space": "cosine"})
#         openai.api_key = self.openai_api_key
# 
#     def __str__(self):
#         """Returns a string representation of the ChromaMemory object."""
#         return f"Chroma Memory | Collection: {self.chroma_collection}"
# 
#     def _get_api_keys(self, config_file):
#         config = configparser.ConfigParser()
#         config.read(config_file)
# 
#         openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
#         chroma_collection = config.get("API_KEYS", "Chroma_Collection")
# 
#         return openai_api_key, chroma_collection
# 
#     def _count_tokens(self, text):
#         tokens = len(self.encoding.encode(text))
#         return tokens
# 
#     def _get_embedding(self, text: str, model: str = "text-embedding-ada-002"):
#         while True:
#             try:
#                 result = openai.Embedding.create(model=model, input=text)
#                 break
#             except (APIError, RateLimitError):
#                 print("OpenAI had an issue, trying again in a few seconds...")
#                 time.sleep(1)
#         return result["data"][0]["embedding"]
# 
#     def create_embeddings_dataframe(self, context_chunks: list):
#         # Calculate embeddings for each chunk with a progress bar
#         embeddings = []
#         progress_bar = tqdm(
#             total=len(context_chunks), desc="Calculating embeddings", position=0
#         )
# 
#         for chunk in context_chunks:
#             embedding = self._get_embedding(chunk)
#             embeddings.append(embedding)
#             progress_bar.update(
#                 1
#             )  # Increment the progress bar after each embedding calculation
#             sys.stdout.flush()
# 
#         progress_bar.close()  # Close the progress bar when the loop is finished
# 
#         # Create the DataFrame with id and chunk columns
#         df = {
#             "id": [f"chunk_{i}" for i in range(len(context_chunks))],
#             "chunk": context_chunks,
#             "embeddings": embeddings
#         }
# 
#         return df
# 
#     def store_single(self, text: str, doc_id: str = None, metadata: dict = None):
#         """Store a single document in Chroma."""
#         assert (
#             self._count_tokens(text) <= 1200
#         ), "Text too long, chunk text before passing to .store_single()"
#         # Compute the embedding
#         embedding = self._get_embedding(text)
#         # Store the document in Chroma
#         if metadata is None:
#             self.collection.add(
#                 documents=[text],
#                 embeddings=[embedding],
#                 ids=[doc_id if doc_id else str(random.randint(1, 10000))],
#             )
#         else:
#             self.collection.add(
#                 documents=[text],
#                 embeddings=[embedding],
#                 metadatas=[metadata],
#                 ids=[doc_id if doc_id else str(random.randint(1, 10000))],
#             )
# 
#     def store(self, context_chunks: list, metadatas=None):
#         """Store multiple documents in Chroma"""
#         data = self.create_embeddings_dataframe(context_chunks)
#         if metadatas is None:
#             self.collection.add(
#                 documents=data["chunk"],
#                 embeddings=data["embeddings"],
#                 ids=data["id"],
#             )
#         else:
#             self.collection.add(
#                 documents=data["chunk"],
#                 embeddings=data["embeddings"],
#                 metadatas=metadatas,
#                 ids=data["id"],
#             )
# 
#     def fetch_context(self, query, top_n=5):
#         
#         if self.collection.count() < 5:
#             top_n = self.collection.count()
#         # Generate the query embedding
#         query_embedding = self._get_embedding(query)
#         # Query the most similar results
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_n,
#         )
#         # Retrieve the documents for the relevant embeddings
#         context_chunks = results['documents']
# 
#         return context_chunks
# =============================================================================

class ChromaMemory:
    def __init__(self, collection_name=None, storage=None):
        if not os.path.exists("config.ini"):
            raise FileNotFoundError("The config file was not found.")

        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        (
            self.openai_api_key,
            self.chroma_collection,
        ) = self._get_api_keys("config.ini")
        
        openai.api_key = self.openai_api_key
        
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-ada-002"
            )
        
        if collection_name:
            self.chroma_collection = collection_name.lower()
            
        self.client = chromadb.Client()
        self.storage = storage
        
        if self.storage:
            self.client = chromadb.PersistentClient(path=self.storage)
        
        try:
            self.collection = self.client.create_collection(self.chroma_collection, embedding_function=self.ef, metadata={"hnsw:space": "cosine"})
        
        except ValueError:
            self.collection = self.client.get_collection(self.chroma_collection)
        
        self.embedder = Embedder()

    def __str__(self):
        """Returns a string representation of the ChromaMemory object."""
        return f"Chroma Memory | Collection: {self.chroma_collection}"

    def _get_api_keys(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
        chroma_collection = config.get("API_KEYS", "Chroma_Collection")

        return openai_api_key, chroma_collection

    def _count_tokens(self, text):
        tokens = len(self.encoding.encode(text))
        return tokens


    def store_single(self, text: str, doc_id: str = None, metadata: dict = None):
        """Store a single document in Chroma."""
        assert (
            self._count_tokens(text) <= 1200
        ), "Text too long, chunk text before passing to .store_single()"
        # Compute the embedding
        embedding = self.embedder.get_embedding(text)
        
        unique_id = doc_id if doc_id else f"chunk_{self.collection.count()}"
        # Store the document in Chroma
        if metadata is None:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[unique_id],
            )
        else:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[unique_id],
            )

    def store(self, context_chunks: list, metadatas=None):
        """Store multiple documents in Chroma"""
        
        start_id = self.collection.count()
        data = asyncio.run(self.embedder.create_embeddings(context_chunks, start_id=start_id))
        if metadatas is None:
            self.collection.add(
                documents=data["chunk"].tolist(),
                embeddings=data["embeddings"].tolist(),
                ids=data["id"].tolist(),  # Convert to list
            )
        else:
            self.collection.add(
                documents=data["chunk"].tolist(),
                embeddings=data["embeddings"].tolist(),
                metadatas=metadatas,
                ids=data["id"].tolist(),  # Convert to list
            )

    def fetch_context(self, query, top_n=5):
        
        if self.collection.count() < 5:
            top_n = self.collection.count()
        # Generate the query embedding
        query_embedding = self.embedder.get_embedding(query)
        # Query the most similar results
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
        )
        # Retrieve the documents for the relevant embeddings
        context_chunks = results['documents']

        return context_chunks