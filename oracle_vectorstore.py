"""
Python Version: 3.11

Description:
    This module provides the class to integrate Oracle Vector DB 
    as Vector Store in llama-index.
"""

import time
from tqdm import tqdm
import array
from typing import List, Any, Dict
from contextlib import contextmanager
import streamlit as st
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from llama_index.core.schema import TextNode, BaseNode

import oracledb
import logging

# Load configurations from the config module
from config import (
    DB_USER,
    DB_PWD,
    DB_HOST_IP,
    DB_SERVICE,
    EMBEDDINGS_BITS,
    ADD_PHX_TRACING
    )

# Phoenix tracing setup if enabled
if ADD_PHX_TRACING:
    from opentelemetry import trace as trace_api
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger("ConsoleLogger")

# Initialize Phoenix tracer if tracing is enabled
tracer = None
if ADD_PHX_TRACING:
    endpoint = "http://127.0.0.1:7777/v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    trace_api.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    tracer = trace_api.get_tracer(__name__)
    OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

@contextmanager
def optional_tracing(span_name):
    """
    Context manager for optional Phoenix tracing.
    
    Args:
        span_name (str): The name of the tracing span.
    
    Yields:
        span: The tracing span context if tracing is enabled, otherwise None.
    """
    if ADD_PHX_TRACING:
        with tracer.start_as_current_span(name=span_name) as span:
            span.set_attribute(OPENINFERENCE_SPAN_KIND, "Retriever")
            span.set_attribute(SpanAttributes.TOOL_NAME, "oracle_vector_store")
            span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, "Oracle DB 23c")
            span.set_status(Status(StatusCode.OK))
            yield span
    else:
        yield None

def oracle_query(embed_query: List[float], top_k: int, verbose=False, approximate=False):
    """
    Executes a query against an Oracle database to find the top_k closest vectors to the given embedding.

    Args:
        embed_query (List[float]): A list of floats representing the query vector embedding.
        top_k (int, optional): The number of closest vectors to retrieve. Defaults to 2.
        verbose (bool, optional): If set to True, additional information about the query and execution time will be printed. Defaults to False.
        approximate (bool, optional): If set to True, use approximate (index) query. Defaults to False.

    Returns:
        VectorStoreQueryResult: Object containing the query results, including nodes, similarities, and ids.
    """
    start_time = time.time()
    DSN = f"{DB_HOST_IP}/{DB_SERVICE}"

    try:
        with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
            with connection.cursor() as cursor:
                array_type = "d" if EMBEDDINGS_BITS == 64 else "f"
                array_query = array.array(array_type, embed_query)
                approx_clause = "APPROXIMATE" if approximate else ""

                select = f"""
                    SELECT V.id, C.CHUNK, C.PAGE_NUM, 
                           VECTOR_DISTANCE(V.VEC, :1, COSINE) AS d,
                           B.NAME 
                    FROM VECTORS V, CHUNKS C, BOOKS B
                    WHERE C.ID = V.ID AND C.BOOK_ID = B.ID
                    ORDER BY d
                    FETCH {approx_clause} FIRST {top_k} ROWS ONLY
                """

                if verbose:
                    logger.info(f"SQL Query: {select}")

                cursor.execute(select, [array_query])
                rows = cursor.fetchall()

                result_nodes, node_ids, similarities = [], [], []

                for row in rows:
                    logger.info(f"{row}")
                    full_clob_data = row[1].read()
                    result_nodes.append(
                        TextNode(
                            id_=row[0],
                            text=full_clob_data,
                            metadata={"file_name": row[4], "page#": row[2]},
                        )
                    )
                    node_ids.append(row[0])
                    similarities.append(row[3])

    except Exception as e:
        logger.error(f"Error occurred in oracle_query: {e}")
        return None

    q_result = VectorStoreQueryResult(
        nodes=result_nodes, similarities=similarities, ids=node_ids
    )

    elapsed_time = time.time() - start_time

    if verbose:
        logger.info(f"Query duration: {round(elapsed_time, 1)} sec.")

    return q_result

def save_embeddings_in_db(embeddings, pages_id, connection):
    """
    Save the provided embeddings to the Oracle database.

    Args:
        embeddings (list): List of embedding vectors.
        pages_id (list): List of page IDs corresponding to the embeddings.
        connection: The Oracle database connection.
    """
    tot_errors = 0

    with connection.cursor() as cursor:
        logger.info("Saving embeddings to DB...")

        for id, vector in zip(tqdm(pages_id), embeddings):
            array_type = "d" if EMBEDDINGS_BITS == 64 else "f"
            input_array = array.array(array_type, vector)

            try:
                cursor.execute("INSERT INTO VECTORS VALUES (:1, :2)", [id, input_array])
            except Exception as e:
                logger.error("Error in save embeddings...")
                logger.error(e)
                tot_errors += 1

    logger.info(f"Total errors in save_embeddings: {tot_errors}")

def save_chunks_in_db(pages_text, pages_id, pages_num, book_id, connection):
    """
    Save the provided text chunks to the Oracle database.

    Args:
        pages_text (list): List of text chunks.
        pages_id (list): List of page IDs.
        pages_num (list): List of page numbers.
        book_id: The book ID to associate with the text chunks.
        connection: The Oracle database connection.
    """
    tot_errors = 0

    with connection.cursor() as cursor:
        logger.info("Saving texts to DB...")
        cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)

        for id, text, page_num in zip(tqdm(pages_id), pages_text, pages_num):
            try:
                cursor.execute(
                    "INSERT INTO CHUNKS (ID, CHUNK, PAGE_NUM, BOOK_ID) VALUES (:1, :2, :3, :4)",
                    [id, text, page_num, book_id],
                )
            except Exception as e:
                logger.error("Error in save chunks...")
                logger.error(e)
                tot_errors += 1

    logger.info(f"Total errors in save_chunks: {tot_errors}")

class OracleVectorStore(VectorStore):
    """
    Class to interface with Oracle DB Vector Store.
    """

    stores_text: bool = True
    verbose: bool = False
    DSN = f"{DB_HOST_IP}/{DB_SERVICE}"

    def __init__(self, verbose=False, enable_hnsw_indexes=False) -> None:
        """
        Initialize OracleVectorStore with optional verbosity and HNSW index support.
        
        Args:
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            enable_hnsw_indexes (bool, optional): Enable HNSW indexes for approximate search. Defaults to False.
        """
        self.verbose = verbose
        self.enable_hnsw_indexes = enable_hnsw_indexes
        self.node_dict: Dict[str, BaseNode] = {}

    def add(self, nodes: List[BaseNode]) -> List[str]:
        """
        Add nodes to the index.

        Args:
            nodes (List[BaseNode]): List of nodes to add.

        Returns:
            List[str]: List of node IDs added.
        """
        ids_list = []
        for node in nodes:
            self.node_dict[node.id_] = node
            ids_list.append(node.id_)

        return ids_list

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes from the index (not implemented).

        Args:
            node_id (str): The ID of the node to delete.
        """
        raise NotImplementedError("Delete not yet implemented for Oracle Vector Store.")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the Oracle DB Vector Store.

        Args:
            query (VectorStoreQuery): The query to execute.

        Returns:
            VectorStoreQueryResult: The query result.
        """
        similarity_top_k = st.session_state['top_k']

        if self.verbose:
            logging.info("---> Calling query on DB with top_k={}".format(similarity_top_k))

        with optional_tracing("oracle_vector_db"):
            return oracle_query(
                query.query_embedding,
                top_k=similarity_top_k,
                verbose=self.verbose,
                approximate=self.enable_hnsw_indexes,
            )

    def persist(self, persist_path=None, fs=None) -> None:
        """
        Persist the VectorStore to the Oracle database.
        """
        if self.node_dict:
            logging.info("Persisting to DB...")

            embeddings = []
            pages_id = []
            pages_text = []
            pages_num = []

            for key, node in self.node_dict.items():
                pages_id.append(node.id_)
                pages_text.append(node.text)
                embeddings.append(node.embedding)
                pages_num.append(node.metadata["page#"])

            with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=self.DSN) as connection:
                save_embeddings_in_db(embeddings, pages_id, connection)
                save_chunks_in_db(pages_text, pages_id, pages_num, book_id=None, connection=connection)
                connection.commit()

            self.node_dict = {}