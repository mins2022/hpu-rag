"""
# hpu-rag
Usinf RAG for Gaudi

This script allows you to fetch and analyze the content of a specified webpage. You can ask questions about the content, including checking for broken links. The script uses a language model (LLM) and Retrieval-Augmented Generation (RAG) setup to process and respond to your queries.

## Setup

1. **Launch your container**

2. **Clone the Repository**

   ```bash
   git clone https://github.com/mins2022/hpu-rag.git
   cd hpu-rag

2. **Run `setup.sh` script** 

   ```bash
   ./setup.sh

3. Set HF_HOME and HF_TOKEN

   ```bash
   # Example 
   export HUGGING_FACE_HUB_TOKEN=hf_xxxxx
   export HF_HOME=/mnt/huggingface/hub

## Usage

1. **Run the Script**

   ```bash
   python llm-rag-web.py

2. Example
   ```bash
   Please enter the URL of the webpage (or type 'quit' to exit): https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html
   Webpage content fetched successfully.
   How can I help you? (or type 'quit' to exit, 'new' for new webpage)
   Is there any broken link?
   In Webpage Document - Broken links detected:https://example.com/broken-link
   Do you want to continue checking the same webpage? (yes/no): yes
   How can I help you? (or type 'quit' to exit, 'new' for new webpage)
   What is the installation guide for drivers?
   In Webpage Document - The installation guide for drivers is as follows...
   Do you want to continue checking the same webpage? (yes/no): no
   Please enter the URL of the webpage (or type 'quit' to exit):
