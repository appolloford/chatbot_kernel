import os
import traceback
import torch
from ipykernel.kernelbase import Kernel
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatbotKernel(Kernel):
    implementation = 'Chatbot'
    implementation_version = '0.1'
    language = 'no-op'
    language_version = '0.1'
    language_info = {
        'name': 'chatbot',
        'mimetype': 'text/plain',
        'file_extension': '.txt',
    }
    banner = "Chatbot kernel - using LLM from huggingface"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_id = None
        self.model = None
        self.conversation = []
        self.cache_dir = os.getenv("HF_HOME", None)

    def _init_llm(self):
        """Initialize a LLM for use"""
        if self.model_id is None:
            raise ValueError("Model ID is not provided!")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map = 'auto',
        )

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):
        try:
            lines = code.split("\n")
            for lidx, line in enumerate(lines):
                if line.strip().startswith('%'):
                    self.handle_magic(line, silent)
                else:
                    # Combine the rest as a single message if no more magics in the front
                    self.handle_chat("\n".join(lines[lidx:]), silent)
                    break

            return {'status': 'ok',
                    # The base class increments the execution count
                    'execution_count': self.execution_count,
                    'payload': [],
                    'user_expressions': {},
                   }

        except Exception as e:
            error_content = {
                'ename': str(type(e)),
                'evalue': str(e),
                'traceback': traceback.format_exc().split('\n')
            }
            self.send_response(self.iopub_socket, 'error', error_content)
            return {'status': 'error', 'execution_count': self.execution_count,
                    'ename': error_content['ename'], 'evalue': error_content['evalue'],
                    'traceback': error_content['traceback']}


    def handle_chat(self, code, silent):
        if self.model is None:
            raise ValueError("Model has not been initialized!")

        if not silent:
            self.conversation.append({"role": "user", "content": code})
            input_ids = self.tokenizer.apply_chat_template(
                self.conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            generated = input_ids
            start = input_ids.shape[-1]
            while 1:
                outputs = self.model.generate(
                    generated,
                    max_new_tokens=8,
                    eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    # use_cache=True,
                    temperature=0.6,
                    top_p=0.9,
                )

                tokens = outputs[0][start:]
                start += len(tokens)
                generated = outputs

                tokens = self.tokenizer.decode(tokens, skip_special_tokens=True)
                stream_content = {'name': 'stdout', 'text': tokens}
                self.send_response(self.iopub_socket, 'stream', stream_content)

                if outputs[0, -1] == self.tokenizer.eos_token_id:
                    break

        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        self.conversation.append({"role": "assistant", "content": response})

        self.send_response(self.iopub_socket, 'clear_output', {'wait': True})
        display_content = {
            'data': {
                'text/markdown': response
            },
            'metadata': {},
            'transient': {'display_id': f'markdown_output_{self.execution_count}'}
        }
        self.send_response(self.iopub_socket, 'display_data', display_content)

    def handle_magic(self, code, silent):
        # Drop the leading '%'
        commands = code[1:].split()
        if len(commands) == 1:
            magic_command = commands[0]
        else:
            magic_command, *magic_argv = commands

        if magic_command == "load":
            self.model_id = magic_argv[0]
            self._init_llm()

        elif magic_command == "new_chat":
            # clean any chat history
            self.conversation = []

        elif magic_command == "hf_home":
            self.cache_dir = magic_argv[0]

        elif magic_command == "model_list":
            if not silent:
                # default cache_dir
                default_home = os.path.join(os.path.expanduser("~"), ".cache")
                HF_HOME = os.path.expanduser(
                    os.getenv(
                        "HF_HOME",
                        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
                    )
                )
                cache_dir = self.cache_dir or HF_HOME
                models = os.listdir(os.path.join(cache_dir, 'hub'))
                output = "\n - ".join(["/".join(m.split('--')[1:]) for m in models if m.startswith("models")])
                output = f"Available models:\n - {output}"
                display_content = {
                    'data': {
                        'text/markdown': output
                    },
                    'metadata': {}
                }
                self.send_response(self.iopub_socket, 'display_data', display_content)

        else:
            raise ValueError(f"Unknown magic keyword: {magic_command}")

