# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:53:53 2023

@author: marca
"""

import openai
import asyncio
import pandas as pd
import configparser
from tqdm.auto import tqdm
import time
from openai.error import RateLimitError, APIError
import tiktoken
import nest_asyncio

nest_asyncio.apply()


class Embedder:
    EMBEDDING_MODEL = "text-embedding-ada-002"

    def __init__(self):
        self.openai_api_key = self._get_api_keys("config.ini")
        openai.api_key = self.openai_api_key

    def _get_api_keys(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
        return openai_api_key

    async def _get_embedding_async(self, text, model):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text, model)

    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL):
        while True:
            try:
                result = openai.Embedding.create(model=model, input=text)
                break
            except (openai.error.APIError, openai.error.RateLimitError):
                print("OpenAI got grumpy, trying again in a few seconds...")
                time.sleep(1)
        return result["data"][0]["embedding"]

    async def create_embeddings(self, context_chunks: list, start_id: int = 0):
        async def embed_chunk(chunk):
            return await self._get_embedding_async(chunk, self.EMBEDDING_MODEL)

        # Initialize progress bar
        progress = tqdm(total=len(context_chunks), desc="Embedding chunks")

        # Create a list of tasks for each chunk to compute embeddings
        tasks = [embed_chunk(chunk) for chunk in context_chunks]

        # Run the tasks asynchronously and gather the results
        embeddings = await asyncio.gather(*tasks)

        # Close the progress bar after completion
        progress.close()

        # Create the DataFrame with id, chunk, and embeddings columns
        df = pd.DataFrame({
            "id": [f"chunk_{start_id + i}" for i in range(len(context_chunks))],
            "chunk": context_chunks,
            "embeddings": embeddings
        })

        return df


