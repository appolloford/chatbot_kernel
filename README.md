# chatbot_kernel
A Jupyter kernel to use Jupyter as a chat window, running downloaded LLMs locally.

## Installation
This kernel package can be install from PyPI
```
$ pip install chatbot_kernel
```

Once the package installed, the kernel spec can be installed to home directory by command:
```
$ python -m chatbot_kernel install --user
```
If you are using virtualenv, do
```
$ python -m chatbot_kernel install --sys-prefix
```
instead.

If you install the package to a virtual environment, you may need to set up `JUPYTER_PATH=/path/to/venv/share/jupyter:$JUPYTER_PATH` so that jupyter can find the kernel

## Usage
A few magics are available in the kernel:
- `%load`: load a pretrained LLM
- `%hf_home`: set the path to find downloaded LLMs, similar to set `HF_HOME` environment variable
- `%model_list`: show the available LLMs
- `%new_chat`: clean up the chat history

Before start chatting, you need to at least download a model from [HuggingFace](https://huggingface.co/docs/hub/models-downloading). For example, `huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct`.
Once models are downloaded, launch a jupyter notebook/lab and execute `%load <model>` and start chatting. Here is an example:
```
%load meta-llama/Meta-Llama-3-8B-Instruct
hi 
who are you
```

## Caveat
Currently, the kernel use the `AutoModelForCausalLM` and it is not supported by all models.
A few models have been tested:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`: needs `sentencepiece` dependency
- `unsloth/llama-3-8b-Instruct-bnb-4bit`: needs `bitsandbytes` dependency

