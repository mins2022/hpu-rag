'''
1. Please login to Hugging Face before running this script.
2. Set your Hugging Face home dir as HF_HOME.
'''

import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import habana_frameworks.torch.core as htcore
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext, VectorStoreIndex, Document

# Set the model name and Hugging Face home directory
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_HOME = os.environ.get('HF_HOME')

if not HF_HOME:
    print("Please set the HF_HOME environment variable.")
    quit()

print(f"Using HF from {HF_HOME}")

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
        torch_dtype=torch.bfloat16,
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

# Function to fetch and parse webpage content
def fetch_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        webpage_content = response.text
        print("Webpage content fetched successfully.")
        return webpage_content
    except Exception as e:
        print(f"Error fetching webpage content: {e}")
        return None

# Function to check for broken links
def check_links(soup, base_url):
    broken_links = []
    for link in soup.find_all('a', href=True):
        url = link['href']
        if not url.startswith('http'):
            url = os.path.join(base_url, url)
        try:
            link_response = requests.head(url, allow_redirects=True)
            if link_response.status_code >= 400:
                broken_links.append(url)
        except requests.RequestException as e:
            broken_links.append(url)
    return broken_links

# Function to extract commands from the webpage content
def extract_commands(soup):
    commands = []
    for code_block in soup.find_all('pre'):
        command = code_block.get_text().strip()
        if command:
            commands.append(command)
    return commands

commands_cache = []

# Interactive query loop
while True:
    # Get the URL of the webpage from the user
    WEBPAGE_URL = input("Please enter the URL of the webpage (or type 'quit' to exit): ")
    if WEBPAGE_URL.lower() == 'quit':
        break

    # Fetch the webpage content
    webpage_content = fetch_webpage_content(WEBPAGE_URL)
    if not webpage_content:
        continue

    # Parse the webpage content
    soup = BeautifulSoup(webpage_content, 'html.parser')
    text_content = soup.get_text()

    # Extract commands from the webpage content
    commands_cache = extract_commands(soup)

    # Create a Document object from the text content and extracted commands
    document_content = f"{text_content}\n\nExtracted Commands:\n" + "\n".join(commands_cache)
    document = Document(text=document_content, metadata={"source": WEBPAGE_URL})

    # Create an index from the document
    index = VectorStoreIndex.from_documents([document])

    # Setup the query engine
    query_engine = index.as_query_engine()

    while True:
        user_input = input("How can I help you? (or type 'quit' to exit, 'new' for new webpage) \n")
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'new':
            break
        if user_input.lower() == 'list commands':
            if commands_cache:
                print("Listing all extracted commands:")
                for command in commands_cache:
                    print(command)
            else:
                print("No commands have been extracted yet. Please enter a webpage URL first.")
        else:
            try:
                response = query_engine.query(user_input)
            except Exception as e:
                response = f"Error during query: {e}"
            print(f"In Webpage Document - {response}")

        # Ask the user if they want to continue with the same webpage
        continue_checking = input("Do you want to continue checking the same webpage? (yes/no): ")
        if continue_checking.lower() != 'yes':
            break