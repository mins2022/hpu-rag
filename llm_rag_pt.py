'''
1. Please login to Hugging Face before running this script.
2. Set your Hugging Face home dir as HF_HOME.
'''

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import habana_frameworks.torch.core as htcore
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext, VectorStoreIndex, download_loader

# Set the model name and Hugging Face home directory
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_HOME = os.environ.get('HF_HOME')

if not HF_HOME:
    print("Please set the HF_HOME environment variable.")
    quit()

print(f"Using HF from {HF_HOME}")

# Define the path to the PDF file
DB_FILE = os.path.join(os.getcwd(), "data/gaudi-install.pdf")

# Load the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=HF_HOME)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    quit()

# Load the model
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        cache_dir=HF_HOME,
        torch_dtype=torch.float16,
        rope_scaling={"type": "dynamic", "factor": 2},
        load_in_8bit=False
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    quit()

# Move the model to HPU and set pad_token_id
try:
    model = model.eval().to("hpu")
    model.config.pad_token_id = model.config.eos_token_id
    print("Model moved to HPU and pad_token_id set.")
except Exception as e:
    print(f"Error moving model to HPU or setting pad_token_id: {e}")
    quit()

# Define the system prompt
system_prompt = """[INST] <>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.<>
"""

# Define the query wrapper prompt
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create the HuggingFaceLLM instance
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer
)

# Create and download embeddings instance
embeddings = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create and set the service context
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
set_global_service_context(service_context)

# Download and load the PDF documents
PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()
documents = loader.load(file_path=Path(DB_FILE), metadata=True)

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Setup the query engine
query_engine = index.as_query_engine()

# Interactive query loop
while True:
    user_input = input("How can I help you? \n")
    if user_input.lower() == 'quit':
        break
    try:
        response = query_engine.query(user_input)
        print(f"In Gaudi Document - {response}")
    except Exception as e:
        print(f"Error during query: {e}")