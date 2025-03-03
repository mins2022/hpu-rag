import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import deepspeed
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext, VectorStoreIndex, Document

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['LOCAL_RANK'] = str(rank)  # Set LOCAL_RANK for DeepSpeed
    initialize_distributed_hpu()  # Initialize HPU distributed environment
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    print(f"Process {rank} initialized.")

def cleanup():
    dist.destroy_process_group()
    print("Process group destroyed.")

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

def get_files(repo_url, file_extensions=None):
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        tree = response.json().get('tree', [])
        files = [
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/{item['path']}"
            for item in tree if file_extensions is None or any(item['path'].endswith(ext) for ext in file_extensions)
        ]
        print(f"Found {len(files)} files.")
        return files
    except Exception as e:
        print(f"Error fetching files: {e}")
        return []

def extract_commands(soup):
    commands = []
    for code_block in soup.find_all('pre'):
        command = code_block.get_text().strip()
        if command:
            commands.append(command)
    return commands

def run(rank, world_size):
    setup(rank, world_size)

    # Set the model name and Hugging Face home directory
    MODEL = "meta-llama/Llama-3.1-70B-Instruct"
    HF_HOME = os.environ.get('HF_HOME')

    if not HF_HOME:
        print("Please set the HF_HOME environment variable.")
        return

    print(f"Using HF from {HF_HOME}")

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=HF_HOME)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load the model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            cache_dir=HF_HOME,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            trust_remote_code=True  # Allow execution of custom code
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # DeepSpeed configuration for inference
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,  # Set to 1 for inference
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True  # Enable bf16
        },
        "zero_optimization": {
            "stage": 3,  # Use ZeRO Stage 3 for maximum memory efficiency
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_scatter": False
        }
    }

    # Initialize DeepSpeed
    try:
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters()
        )
        print("DeepSpeed initialized successfully.")
    except Exception as e:
        print(f"Error initializing DeepSpeed: {e}")
        return

    try:
        model_engine.eval().to("hpu")
        model_engine.module.config.pad_token_id = model_engine.module.config.eos_token_id
        print("Model moved to HPU and pad_token_id set.")
    except Exception as e:
        print(f"Error moving model to HPU or setting pad_token_id: {e}")
        return

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
        context_window=512,  # Further reduce context window to save memory
        max_new_tokens=16,  # Further reduce to save memory
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        model=model_engine.module,
        tokenizer=tokenizer
    )

    # Create and download embeddings instance
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Create and set the service context
    service_context = ServiceContext.from_defaults(
        chunk_size=128,  # Further reduce to manage memory usage
        llm=llm,
        embed_model=embeddings
    )
    set_global_service_context(service_context)

    commands_cache = []

    # Interactive query loop
    while True:
        # Get the URL of the GitHub repository from the user
        GITHUB_REPO_URL = input("Please enter the URL of the GitHub repository (or type 'quit' to exit): ")
        if GITHUB_REPO_URL.lower() == 'quit':
            break

        # Get all .py, README.md, and .txt files in the repository
        file_links = get_files(GITHUB_REPO_URL, file_extensions=['.py', 'README.md', '.txt'])
        if not file_links:
            print("No files found.")
            continue

        # List all files
        print("Found files:")
        for link in file_links:
            print(link)

        # Ask the user if they want to process these files
        process_files = input("Do you want to process these files? (yes/no): ")
        if process_files.lower() != 'yes':
            continue

        documents = []
        for file_url in file_links:
            file_content = fetch_webpage_content(file_url)
            if not file_content:
                continue
            file_soup = BeautifulSoup(file_content, 'html.parser')
            text_content = file_soup.get_text()

            # Extract commands from the file content
            commands_cache = extract_commands(file_soup)

            # Create a Document object from the text content and extracted commands
            document_content = f"{text_content}\n\nExtracted Commands:\n" + "\n".join(commands_cache)
            document = Document(text=document_content, metadata={"source": file_url})
            documents.append(document)

        # Create an index from the documents
        index = VectorStoreIndex.from_documents(documents)

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

    cleanup()

if __name__ == "__main__":
    world_size = 8  # Number of devices available
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)