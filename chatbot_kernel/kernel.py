import traceback
import accelerate
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

    def _init_llm(self):
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

    def chat(self, message):
        if self.model is None:
            raise ValueError("Model has not been initialized!")

        self.conversation.append({"role": "user", "content": message})
        input_ids = self.tokenizer.apply_chat_template(
            self.conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        self.conversation.append({"role": "assistant", "content": response})
        return response

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):
        try:
            if code.startswith("%load"):
                self.model_id = code.lstrip("%load").strip()
                self._init_llm()
    
            elif code.startswith("%newchat"):
                # remove all conversation except the system message
                self.conversation = [message for message in self.conversation if message.get("role") == "system"]

            # Only valid for llama model
            # elif code.startswith("%system"):
            #     self.conversation = [{"role": "system", "content": code.lstrip("%system").strip()}]

            elif not silent:
                response = self.chat(code)
                display_content = {
                    'data': {
                        'text/markdown': response
                    },
                    'metadata': {}
                }
                self.send_response(self.iopub_socket, 'display_data', display_content)
                # stream_content = {'name': 'stdout', 'text': chat(code)}
                # self.send_response(self.iopub_socket, 'stream', stream_content)
    
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
