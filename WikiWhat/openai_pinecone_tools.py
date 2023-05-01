# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:21:16 2023

@author: marca
"""


from wiki_tools import *
import time
import pandas as pd
import numpy as np
import openai
from openai.error import RateLimitError
import pinecone
from pinecone import PineconeProtocolError
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import threading
import nest_asyncio

nest_asyncio.apply()

openai.api_key = OPENAI_API_KEY


def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    while True:
        try:
            result = openai.Embedding.create(model=model, input=text)
            break
        except (APIError, RateLimitError):
            print("OpenAI got grumpy, trying again in a few seconds...")
            time.sleep(10)
    return result["data"][0]["embedding"]


# This function is a helper function that runs the get_embedding function asynchronously.
async def get_embedding_async(text, model):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embedding, text, model)


# This function computes the document embeddings for a given dataframe containing Wikipedia segments.
async def compute_doc_embeddings(df: pd.DataFrame, model: str = EMBEDDING_MODEL):
    # This helper function creates an embedding for a given row and updates the progress bar.
    async def embed_row(row, progress):
        text = row["title"] + " " + row["heading"] + " " + row["content"]
        embedding = await get_embedding_async(text, model)
        progress.update(1)
        return embedding

    # Initialize the progress bar for displaying the progress of the embedding process.
    progress = tqdm(total=df.shape[0], desc="Embedding segments")
    # Create a list of tasks for each row in the dataframe to compute embeddings.
    tasks = [embed_row(row, progress) for _, row in df.iterrows()]
    # Run the tasks asynchronously and gather the results.
    embeddings = await asyncio.gather(*tasks)
    # Close the progress bar after completion.
    progress.close()

    # Create a new dataframe with the computed embeddings.
    embedding_columns = {
        f"embedding{idx}": [embedding[idx] for embedding in embeddings]
        for idx in range(len(embeddings[0]))
    }
    embedding_df = pd.DataFrame(embedding_columns)
    # Concatenate the original dataframe with the embeddings dataframe.
    df = pd.concat([df, embedding_df], axis=1)

    return df


def create_dataframe(pages, output_filename=None):
    res = []
    for page in pages:
        print(page.title)
        res += extract_sections(page.content, page.title)

    df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens"])
    df = df[df.tokens > 40]
    df = df.drop_duplicates(["title", "heading"])
    df = df.reset_index().drop("index", axis=1)
    df = asyncio.run(compute_doc_embeddings(df))

    if output_filename:
        df.to_csv(output_filename, index=False)

    return df


### PINECONE FUNCTIONS ###


def store_embeddings_in_pinecone(
    index=PINECONE_INDEX,
    namespace=PINECONE_NAMESPACE,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
    csv_filepath=None,
    topic_name=None,
    dataframe=None,
):
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Instantiate Pinecone's Index
    pinecone_index = pinecone.Index(index_name=index)

    # Check if a dataframe is provided and is not empty
    if dataframe is not None and not dataframe.empty:
        batch_size = 80
        vectors_to_upsert = []
        batch_count = 0
        topic_name = f"wiki_{topic_name}"

        # Calculate the total number of batches
        total_batches = -(-len(dataframe) // batch_size)

        # Create a tqdm progress bar object
        progress_bar = tqdm(total=total_batches, desc="Loading info into Pinecone")

        # Iterate through each row in the dataframe
        for index, row in dataframe.iterrows():
            context_chunk = row["content"]

            # Create a vector from the embeddings
            vector = [float(row[f"embedding{i}"]) for i in range(1536)]

            # Create an index for Pinecone
            idx = f"wiki_{index}"

            # Prepare metadata for upsert
            metadata = {"topic_name": topic_name, "context": context_chunk}
            vectors_to_upsert.append((idx, vector, metadata))

            # Upsert when the batch is full or it's the last row
            if len(vectors_to_upsert) == batch_size or index == len(dataframe) - 1:
                while True:
                    try:
                        # Upsert the batch of vectors to Pinecone
                        upsert_response = pinecone_index.upsert(
                            vectors=vectors_to_upsert, namespace=namespace
                        )

                        batch_count += 1
                        vectors_to_upsert = []

                        # Update the progress bar
                        progress_bar.update(1)
                        break

                    except pinecone.core.client.exceptions.ApiException:
                        print(
                            "Pinecone is a little overwhelmed, trying again in a few seconds..."
                        )
                        time.sleep(10)

        # Close the progress bar after completing all upserts
        progress_bar.close()

    else:
        print("No dataframe to retrieve embeddings")


def fetch_context_from_pinecone(
    query,
    top_n=5,
    index=PINECONE_INDEX,
    namespace=PINECONE_NAMESPACE,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
):
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Generate the query embedding
    query_embedding = get_embedding(query)

    # Instantiate Pinecone's Index
    pinecone_index = pinecone.Index(index_name=index)

    # Try querying Pinecone for the most similar embeddings until successful
    while True:
        try:
            # Query Pinecone with the query_embedding, asking for top_n matches
            query_response = pinecone_index.query(
                namespace=namespace,
                top_k=top_n,
                include_values=False,
                include_metadata=True,
                vector=query_embedding,
            )
            # Break the loop if the query is successful
            break

        except PineconeProtocolError:
            print("Pinecone needs a moment....")
            time.sleep(3)
            continue

    # Retrieve metadata for the relevant embeddings
    context_chunks = [
        match["metadata"]["context"] for match in query_response["matches"]
    ]

    return context_chunks


def check_topic_exists_in_pinecone(
    topic_name: str,
    index: str = PINECONE_INDEX,
    namespace: str = PINECONE_NAMESPACE,
    pinecone_api_key: str = PINECONE_API_KEY,
    pinecone_env: str = PINECONE_ENV,
) -> bool:
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    topic_name = f"wiki_{topic_name}"
    metadata_filter = {"topic_name": {"$eq": topic_name}}

    index = pinecone.Index(index)
    query_embedding = get_embedding(topic_name)

    query_response = index.query(
        namespace=namespace,
        top_k=1,
        include_values=False,
        include_metadata=True,
        filter=metadata_filter,
        vector=query_embedding,
    )

    return len(query_response["matches"]) != 0


### END PINECONE


def generate_response(
    messages,
    model="gpt-3.5-turbo",
    temperature=0.5,
    n=1,
    max_tokens=4000,
    frequency_penalty=0,
):
    token_ceiling = 4096
    if model == "gpt-4":
        max_tokens = 8000
        token_ceiling = 8000
    # Calculate the number of tokens in the messages
    tokens_used = sum([count_tokens(msg["content"]) for msg in messages])
    tokens_available = token_ceiling - tokens_used

    # Adjust max_tokens to not exceed the available tokens
    max_tokens = min(max_tokens, (tokens_available - 100))

    # Reduce max_tokens further if the total tokens exceed the model limit
    if tokens_used + max_tokens > token_ceiling:
        max_tokens = token_ceiling - tokens_used - 10

    if max_tokens < 1:
        max_tokens = 1

    # Generate a response
    max_retries = 10
    retries = 0
    while True:
        if retries < max_retries:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                )
                break
            except (RateLimitError, KeyboardInterrupt):
                time.sleep(60)
                retries += 1
                print("Server overloaded, retrying in a minute")
                continue
        else:
            print("Failed to generate prompt after max retries")
            return
    response = completion.choices[0].message.content
    return response
