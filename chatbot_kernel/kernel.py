import os
import traceback
import torch
from ipykernel.kernelbase import Kernel
from transformers import AutoTokenizer, AutoModelForCausalLM


class ChatbotKernel(Kernel):
    implementation = "Chatbot"
    implementation_version = "0.1"
    language = "no-op"
    language_version = "0.1"
    language_info = {
        "name": "chatbot",
        "mimetype": "text/plain",
        "file_extension": ".txt",
    }
    banner = "Chatbot kernel - using LLM from huggingface"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_id = None
        self.model = None
        self.conversation = []
        self.cache_dir = os.getenv("HF_HOME", None)

        self.magic_commands = {
            "help": {
                "description": "Show this help message",
                "action": self.handle_help_magic,
            },
            "load": {
                "description": "Load a pre-trained model to start chatting",
                "action": self.handle_load_magic,
            },
            "new_chat": {
                "description": "Start a new chat",
                "action": self.handle_new_chat_magic,
            },
            "hf_home": {
                "description": "Set the path to models, override <code>HF_HOME</code> from the environment",
                "action": self.handle_hf_home_magic,
            },
            "model_list": {
                "description": "List all available models",
                "action": self.handle_model_list_magic,
            },
        }

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        """Execute user code. Inherit from `ipykernel.kernelbase.Kernel`."""
        try:
            lines = code.split("\n")
            for lidx, line in enumerate(lines):
                if line.strip().startswith("%"):
                    self.handle_magic(line, silent)
                else:
                    # Combine the rest as a single message if no more magics in the front
                    self.handle_chat("\n".join(lines[lidx:]), silent)
                    break

            return {
                "status": "ok",
                # The base class increments the execution count
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }

        except Exception as e:
            error_content = {
                "ename": str(type(e)),
                "evalue": str(e),
                "traceback": traceback.format_exc().split("\n"),
            }
            self.send_response(self.iopub_socket, "error", error_content)
            return {
                "status": "error",
                "execution_count": self.execution_count,
                "ename": error_content["ename"],
                "evalue": error_content["evalue"],
                "traceback": error_content["traceback"],
            }

    def handle_chat(self, code, silent):
        """Handle input for normal chat."""
        if self.model is None:
            raise ValueError("Model has not been initialized!")

        if not silent:
            # Append new user input into conversation
            self.conversation.append({"role": "user", "content": code})
            input_ids = self.tokenizer.apply_chat_template(
                self.conversation, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            generated = input_ids
            start = input_ids.shape[-1]
            while 1:
                # Request new tokens until LLM stop generating
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

                # Streaming output
                tokens = self.tokenizer.decode(tokens, skip_special_tokens=True)
                stream_content = {"name": "stdout", "text": tokens}
                self.send_response(self.iopub_socket, "stream", stream_content)

                if outputs[0, -1] == self.tokenizer.eos_token_id:
                    break

        # Append the chatbot response into conversation
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        self.conversation.append({"role": "assistant", "content": response})

        # Clean the plain text stream output
        self.send_response(self.iopub_socket, "clear_output", {"wait": True})

        # Re-render the response in markdown
        display_content = {
            "data": {"text/markdown": response},
            "metadata": {},
            "transient": {"display_id": f"markdown_output_{self.execution_count}"},
        }
        self.send_response(self.iopub_socket, "display_data", display_content)

    def handle_magic(self, code, silent):
        """Handle magic commands."""
        # Drop the leading '%'
        commands = code[1:].split()
        magic_command, *magic_argv = commands

        if magic_command in self.magic_commands.keys():
            action = self.magic_commands.get(magic_command).get("action")
            action(*magic_argv)
        else:
            raise ValueError(f"Unknown magic keyword: {magic_command}")

    def handle_help_magic(self):
        """Handle `%help` magic command."""
        help_table_row = [
            "<table style='border-collapse: collapse; width: 100%;'>",
            "<colgroup><col style='width: 20%'><col style='width: 80%'></colgroup>",
            "<tr><th>Method</th>",
            "<th>Description</th></tr>",
        ]

        for magic_command, magic_dict in self.magic_commands.items():
            magic_desc = magic_dict.get("description")
            help_table_row.append(f"<tr><td style='text-align: left;'>{magic_command}</td>")
            help_table_row.append(f"<td style='text-align: left;'>{magic_desc}</td></tr>")

        help_table_row.append("</table>")
        help_table = "\n".join(help_table_row)
        display_content = {
            "data": {"text/markdown": help_table},
            "metadata": {},
        }
        self.send_response(self.iopub_socket, "display_data", display_content)

    def handle_load_magic(self, *args):
        """Handle `%load` magic command."""
        self.model_id = args[0]

        if self.model_id is None:
            raise ValueError("Model ID is not provided!")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def handle_new_chat_magic(self):
        """Handle `%new_chat` magic command."""
        # Clean any chat history
        self.conversation = []

    def handle_hf_home_magic(self, *args):
        """Handle `%hf_home` magic command."""
        self.cache_dir = args[0]

    def handle_model_list_magic(self):
        """Handle `%model_list` magic command."""
        # Get default cache_dir
        default_home = os.path.join(os.path.expanduser("~"), ".cache")
        HF_HOME = os.path.expanduser(
            os.getenv(
                "HF_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
            )
        )

        cache_dir = self.cache_dir or HF_HOME

        # Parse directories in the cache_dir
        models = os.listdir(os.path.join(cache_dir, "hub"))
        output = "\n - ".join(["/".join(m.split("--")[1:]) for m in models if m.startswith("models")])
        output = f"Available models:\n - {output}"
        display_content = {"data": {"text/markdown": output}, "metadata": {}}
        self.send_response(self.iopub_socket, "display_data", display_content)
