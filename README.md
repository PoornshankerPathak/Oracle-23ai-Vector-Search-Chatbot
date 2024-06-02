# Integrate Oracle 23ai Vector DB and OCI GenAI with Llama-index (v.0.10+)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## The UI of the **Knowledge Assistant** you can build using following examples.

![screenshot](./screenshot.png)

This repository contains all the work done on the development of RAG applications using:

* [Oracle AI Vector Search](https://www.oracle.com/news/announcement ocw-integrated-vector-database-augments-generative-ai-2023-09-19/)
* Oracle OCI [GenAI Service](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/generative-ai/home.htm)
* Oracle OCI[ Embeddings](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/generative-ai/embed-models.htm)
* Cohere Reranking
* Cohere Reranker
* OCI [ADS 2.10.0](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/large_language_model/langchain_models.html) (with support for OCI GenAI)
* [llama-index](https://docs.llamaindex.ai/en/stable/)


## What is RAG?

A very good introduction to what **Retrieval Augmented Generation** (RAG) is can be found [here](https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/)


## Setup

* configure conda environment based on requirement.txt
* configure .oci directory with .key& config for [OCI authentication](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm).
* update parameter in config.py

## execute chatbot
* streamlit run app.py

## this work is inspired by https://github.com/luigisaetta/llamaindex10_oracle