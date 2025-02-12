# hpu-rag
Using RAG for Gaudi

This script fetches and analyzes the content of a specified webpage, allowing you to ask questions about the content. It utilizes a language model (LLM) and a Retrieval-Augmented Generation (RAG) setup to process and respond to your queries, demonstrating a simple example of RAG capabilities.

## Setup

1. **Launch your container**

2. **Clone the Repository**

   ```bash
   git clone https://github.com/mins2022/hpu-rag.git
   cd hpu-rag

2. **Run `setup.sh` script** 

   ```bash
   bash ./setup.sh

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
   Please enter the URL of the webpage (or type 'quit' to exit): https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html#installing-deepspeed-library
   Webpage content fetched successfully.

   How can I help you? (or type 'quit' to exit, 'new' for new webpage)

   Is there any broken link?
   
   In Webpage Document - Broken links detected:
   https://github.com/HabanaAI/Model-References/tree/1.19.2/PyTorch/nlp/DeepSpeedExamples/deepspeed-bert
   https://github.com/HabanaAI/Model-References/blob/1.19.2/PyTorch/nlp/DeepSpeedExamples/deepspeed-bert/modeling.py
   https://github.com/HabanaAI/Megatron-DeepSpeed/tree/1.19.2/megatron/model/gpt_model.py
   https://github.com/HabanaAI/Model-References/blob/1.19.2/PyTorch/nlp/DeepSpeedExamples/deepspeed-bert/run_pretraining.py
   
   Do you want to continue checking the same webpage? (yes/no): yes
   How can I help you? (or type 'quit' to exit, 'new' for new webpage)

   can you provide ZeRO-1 usage example in this webpage?
  
   In Webpage Document -  
   The usage example for ZeRO-1 is not explicitly provided in the webpage.
   However, it mentions that ZeRO-1 is a configuration that partitions the optimizer states across the ranks so that each process updates its own partition. For further details, refer to Using ZeRO section.
   
   However, the webpage does provide an example of ZeRO-2 configuration, which is a more advanced version of ZeRO-1. Here is the example:
   
   "zero_optimization": {
    "stage": 2,
   ...
   }
 
   This example is for ZeRO-2, which is an extension of ZeRO-1. If you want to use ZeRO-1, you would need to set the "stage" parameter to 1 instead of 2.
   However, the webpage does not provide a specific example for ZeRO-1.

   Do you want to continue checking the same webpage? (yes/no): yes
   How can I help you? (or type 'quit' to exit, 'new' for new webpage) 

   But there is a ZeRO-3 example in the website. is this mnore advanded version than ZeRO-2?
   Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
   In Webpage Document -  
   The text does not provide a direct comparison between ZeRO-2 and ZeRO-3.
   However, it does mention that ZeRO-3 is a more advanced version that automatically collects and partitions the model state during the forward and backward passes, which is not mentioned for ZeRO-2.
   Additionally, ZeRO-3 is recommended to use only optimizers that have been tested with DeepSpeed ZeRO, which may imply that ZeRO-3 is a more complex and advanced configuration.
   Therefore, it can be inferred that ZeRO-3 is likely a more advanced version than ZeRO-2.
