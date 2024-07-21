import os
import traceback
import torch
from ipykernel.kernelbase import Kernel
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatbotKernelConfig:
    def __init__(self):
        self._dtype = torch.bfloat16
        self._temperature = 0.6
        self._n_predict = 1000
        self._n_new_tokens = 8

    @classmethod
    def _attributes(cls):
        return [a for a in cls.__dict__.keys() if not a.startswith("_")]

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype_str):
        dtype_mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if dtype_str not in dtype_mapping.keys():
            raise ValueError(f"Invalid dtype. Choose one from {dtype_mapping.keys()}")
        self._dtype = dtype_mapping.get(dtype_str)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temp):
        try:
            temp = float(temp)
            if temp < 0 or temp > 1:
                raise ValueError
            self._temperature = temp
        except:
            raise ValueError(f"`temperature` must be a float between 0 and 1, but `{temp}` is got.")

    @property
    def n_predict(self):
        return self._n_predict

    @n_predict.setter
    def n_predict(self, n_pred):
        try:
            n_pred = int(n_pred)
            self._n_predict = n_pred
        except:
            raise ValueError(f"`n_predict` must be a integer, but `{n_pred}` is got.")

    @property
    def n_new_tokens(self):
        return self._n_new_tokens

    @n_new_tokens.setter
    def n_new_tokens(self, n_tokens):
        try:
            n_tokens = int(n_tokens)
            self._n_new_tokens = n_tokens
        except:
            raise ValueError(f"`n_new_tokens` must be a integer, but `{n_tokens}` is got.")


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

        self.chatbot_config = ChatbotKernelConfig()
        self.magic_commands = {
            "help": {
                "description": "Show this help message",
                "action": self._handle_help_magic,
            },
            "load": {
                "description": "Load a pre-trained model to start chatting",
                "action": self._handle_load_magic,
            },
            "new_chat": {
                "description": "Start a new chat",
                "action": self._handle_new_chat_magic,
            },
            "hf_home": {
                "description": "Set the path to models, override <code>HF_HOME</code> from the environment",
                "action": self._handle_hf_home_magic,
            },
            "model_list": {
                "description": "List all available models",
                "action": self._handle_model_list_magic,
            },
            "config": {
                "description": "Set advanced configuration. `%config help` to see options",
                "action": self._handle_config,
            },
        }

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        """Execute user code. Inherit from `ipykernel.kernelbase.Kernel`."""
        try:
            lines = code.split("\n")
            for lidx, line in enumerate(lines):
                if line.strip().startswith("%"):
                    self._handle_magic(line)
                else:
                    # Combine the rest as a single message if no more magics in the front
                    self._handle_chat("\n".join(lines[lidx:]), silent)
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

    def _handle_chat(self, code, silent):
        """Handle input for normal chat. Invoked by `do_execute`

        Args:
            code (str): Input text
            silent (bool): If True, send response back to the cell
        """
        if self.model is None:
            raise ValueError(
                "Model has not been initialized! Use `%load` to load a model before starting chat. "
                "Check more options in `%help`"
            )

        if not silent:
            # Append new user input into conversation
            self.conversation.append({"role": "user", "content": code})
            input_ids = self.tokenizer.apply_chat_template(
                self.conversation, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            generated = input_ids
            start = input_ids.shape[-1]
            count = 0
            n_predict = self.chatbot_config.n_predict
            while n_predict == -1 or count < n_predict:
                count += 1
                # Request new tokens until LLM stop generating
                outputs = self.model.generate(
                    generated,
                    max_new_tokens=self.chatbot_config.n_new_tokens,
                    eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    # use_cache=True,
                    temperature=self.chatbot_config.temperature,
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

    def _handle_config(self, *args):
        """Handle `%config` magic command. Invoked by `_handle_magic`

        Args:
            args (list): Target option and value
        """
        help_options = {
            "help": "Show this option table",
            "temperature": "The creativity of LLMs. Default: 0.6",
            "dtype": "The data type of LLMs. Model needs to be reloaded. Default: bfloat16",
            "n_predict": "The max number of token prediction. If -1, the response only stop when no more tokens are "
                         "generated. Default: 1000",
            "n_new_tokens": "The number of new tokens printed on the screen at one time. "\
                            "`n_new_tokens` * `n_predict` will be the max length of one response. Default: 8"
        }

        config_key, *config_value = args
        if config_key == "help":
            help_message = "Usage: %config \<option\> \<value\>"
            help_table_row = [
                "<table style='border-collapse: collapse; width: 100%;'>",
                "<colgroup><col style='width: 20%'><col style='width: 80%'></colgroup>",
                "<tr><th>Options</th>",
                "<th>Description</th></tr>",
            ]

            for option, desc in help_options.items():
                help_table_row.append(f"<tr><td style='text-align: left;'>{option}</td>")
                help_table_row.append(f"<td style='text-align: left;'>{desc}</td></tr>")

            help_table_row.append("</table>")
            help_message = "\n".join([help_message, *help_table_row])
            display_content = {
                "data": {"text/markdown": help_message},
                "metadata": {},
            }
            self.send_response(self.iopub_socket, "display_data", display_content)
        elif config_key in self.chatbot_config._attributes():
            if len(config_value) == 0:
                raise RuntimeError(f"Keyword {config_key} is provided but value is not provided")
            elif len(config_value) > 1:
                raise RuntimeError(f"To many values {config_value} for {config_key}")

            config_value = config_value[0]
            setattr(self.chatbot_config, config_key, config_value)
        else:
            raise ValueError(f"Unknow config keyword: {config_key}")

    def _handle_magic(self, code):
        """Handle magic commands. Invoked by `do_execute`

        Args:
            code (str): plain text of magic command and its arguments
        """
        # Drop the leading '%'
        commands = code[1:].split()
        magic_command, *magic_argv = commands

        if magic_command in self.magic_commands.keys():
            action = self.magic_commands.get(magic_command).get("action")
            action(*magic_argv)
        else:
            raise MagicError(f"Unknown magic keyword: {magic_command}. Please check `%help`")

    def _handle_help_magic(self):
        """Handle `%help` magic command. Invoked by `_handle_magic`"""
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

    def _handle_load_magic(self, *args):
        """Handle `%load` magic command. Invoked by `_handle_magic`

        Args:
            args (list): The model to be loaded. Only use the first position
        """
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
            torch_dtype=self.chatbot_config.dtype,
            device_map="auto",
            # local_files_only=True,
        )

    def _handle_new_chat_magic(self):
        """Handle `%new_chat` magic command. Invoked by `_handle_magic`"""
        # Clean any chat history
        self.conversation = []

    def _handle_hf_home_magic(self, *args):
        """Handle `%hf_home` magic command. Invoked by `_handle_magic`

        Args:
            args (list): The target HF_HOME path. Only use the first position
        """
        self.cache_dir = args[0]

    def _handle_model_list_magic(self):
        """Handle `%model_list` magic command. Invoked by `_handle_magic`"""
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
