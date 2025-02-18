import os
import traceback
import torch
from transformers import AutoConfig
from ipykernel.kernelbase import Kernel
from urllib.parse import urlparse
from vllm import LLM, SamplingParams


class ChatbotKernelConfig:
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    def __init__(self):
        self._dtype = torch.bfloat16
        self._temperature = 0.6
        self._n_images = 4
        self._n_predict = 1000
        self._n_new_tokens = 8
        self._tensor_parallel_size = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))

    @classmethod
    def _attributes(cls):
        return [a for a in cls.__dict__.keys() if not a.startswith("_")]

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype_str):
        if dtype_str not in self.dtype_mapping.keys():
            raise ValueError(f"Invalid dtype. Choose one from {self.dtype_mapping.keys()}")
        self._dtype = self.dtype_mapping.get(dtype_str)

    @property
    def n_images(self):
        return self._n_images

    @n_images.setter
    def n_images(self, n_images):
        try:
            n_images = int(n_images)
            if n_images < 1:
                raise ValueError
            self._n_images = n_images
        except:
            raise ValueError(f"`n_images` must be a positive integer, but `{n_images}` is got.")

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
    def tensor_parallel_size(self):
        return self._tensor_parallel_size

    @tensor_parallel_size.setter
    def tensor_parallel_size(self, tensor_parallel_size: str):
        try:
            tensor_parallel_size = int(tensor_parallel_size)
            if tensor_parallel_size < 0:
                raise ValueError
            self._tensor_parallel_size = tensor_parallel_size
        except:
            raise ValueError(f"`tensor_parallel_size` must be a positive integer, but `{tensor_parallel_size}` is got.")


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
        self.images = []

        # Get default cache_dir
        default_home = os.path.join(os.path.expanduser("~"), ".cache")
        HF_HOME = os.path.expanduser(
            os.getenv(
                "HF_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
            )
        )
        self.cache_dir = HF_HOME

        self.chatbot_config = ChatbotKernelConfig()
        self.magic_commands = {
            "help": {
                "description": "Show this help message",
                "action": self._handle_help_magic,
            },
            "image": {
                "description": "The input images for multimodel models, provided by URL or local filepath",
                "action": self._handle_image_magic,
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
                "Model has not been initialized! Use `%load` to load a model before starting chat. Check more options"
                " in `%help`"
            )

        config = AutoConfig.from_pretrained(self.model_id)
        is_vision = getattr(config, "vision_config", None) is not None

        # vision models do not support continue_final_message
        continue_final_message = not is_vision

        if not silent:
            # Append new user input into conversation
            content = []
            for image in self.images:
                content.append({"type": "image_url", "image_url": {"url": image}})
            content.append({"type": "text", "text": code})
            self.images = []  # clean images once they have be saved in conversation
            self.conversation.append({"role": "user", "content": content})

            sampling_params = SamplingParams(
                temperature=self.chatbot_config.temperature,
                max_tokens=self.chatbot_config.n_new_tokens,
                top_p=0.9,
            )

            response = ""
            count = 0
            n_predict = self.chatbot_config.n_predict
            while n_predict == -1 or count < n_predict:
                count += 1
                # Update assistant's response and request for new tokens to accomplish it
                conversation = [
                    *self.conversation,
                    {"role": "assistant", "content": [{"type": "text", "text": response}]},
                ]
                # Request new tokens until LLM stop generating
                output = self.model.chat(
                    conversation,
                    sampling_params,
                    use_tqdm=False,
                    add_generation_prompt=False,
                    continue_final_message=continue_final_message,
                )

                generated = output[0].outputs[0].text
                generated = generated.lstrip("<|start_header_id|>assistant<|end_header_id|>")
                response = "".join([response, generated])

                # Streaming output
                stream_content = {"name": "stdout", "text": generated}
                self.send_response(self.iopub_socket, "stream", stream_content)

                # vision models do not support continue_final_message, quit after one iteration
                if output[0].outputs[0].finish_reason == "stop" or vision_config:
                    break

        # Append the chatbot response into conversation
        self.conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

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
        dtype_options_str = ", ".join(self.chatbot_config.dtype_mapping.keys())
        help_options = {
            "help": "Show this option table",
            "show": "Display the currect configurations",
            "dtype": (
                f"The data type of LLMs. Model needs to be reloaded. Default: bfloat16. Options: {dtype_options_str}"
            ),
            "n_images": "The max number of images to be handled by vision models. Default: 4",
            "n_predict": (
                "The max number of token prediction. If -1, the response only stop when no more tokens are generated."
                " Default: 1000"
            ),
            "n_new_tokens": (
                "The number of new tokens printed on the screen at one time. `n_new_tokens` * `n_predict` will be the"
                " max length of one response. Default: 8"
            ),
            "temperature": "The creativity of LLMs. Default: 0.6",
            "tensor_parallel_size": (
                "Number of tensor parallel replicas. Default to `SLURM_GPUS_ON_NODE` if the environment variable is"
                " set, otherwise 1"
            ),
        }

        config_key, *config_value = args
        if config_key in ["help", "show"]:
            if config_key == "help":
                column_label = "Description"
                message = "Usage: %config \<option\> \<value\>"
            else:
                column_label = "Value"
                message = ""

            config_table_row = [
                f"<table style='border-collapse: collapse; width: 100%;'>",
                f"<colgroup><col style='width: 20%'><col style='width: 80%'></colgroup>",
                f"<tr><th>Options</th>",
                f"<th>{column_label}</th></tr>",
            ]

            for option, desc in help_options.items():
                if config_key == "help":
                    value = desc
                elif config_key == "show" and option not in ["help", "show"]:
                    value = getattr(self.chatbot_config, option)
                else:
                    continue
                config_table_row.append(f"<tr><td style='text-align: left;'>{option}</td>")
                config_table_row.append(f"<td style='text-align: left;'>{value}</td></tr>")

            config_table_row.append("</table>")
            message = "\n".join([message, *config_table_row])
            display_content = {
                "data": {"text/markdown": message},
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

        if config_key in ["dtype", "n_images", "tensor_parallel_size"]:
            self._load_model()

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
            raise ValueError(f"Unknown magic keyword: {magic_command}. Please check `%help`")

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

    def _handle_image_magic(self, *args):
        """Handle `%image` magic command. Invoked by `_handle_magic`

        Args:
            args (list): The images to be loaded.
        """
        for image_url in args:
            url_parsed = urlparse(image_url)
            if url_parsed.scheme == "":
                self.images.append("".join(["file://", url_parsed.path]))
            else:
                self.images.append(url_parsed.geturl())

    def _handle_load_magic(self, *args):
        """Handle `%load` magic command. Invoked by `_handle_magic`

        Args:
            args (list): The model to be loaded. Only use the first position
        """
        model_id = args[0]

        if model_id is None:
            raise ValueError("Model ID is not provided!")

        try:
            stream_content = {"name": "stdout", "text": f"Looking for local model: {model_id} ...\n"}
            self.send_response(self.iopub_socket, "stream", stream_content)

            model_base_path = os.path.join(self.cache_dir, "hub", "--".join(["models", *model_id.split("/")]))
            snapshot = open(os.path.join(model_base_path, "refs", "main")).read()
            self.model_id = os.path.join(model_base_path, "snapshots", snapshot)
        except:
            stream_content = {
                "name": "stdout",
                "text": f"Cannot find {model_id} in {self.cache_dir}. Try downloading ...\n",
            }
            self.send_response(self.iopub_socket, "stream", stream_content)

            self.model_id = model_id

        stream_content = {"name": "stdout", "text": f"Loading {self.model_id} ...\n"}
        self.send_response(self.iopub_socket, "stream", stream_content)

        self._load_model()

    def _handle_new_chat_magic(self):
        """Handle `%new_chat` magic command. Invoked by `_handle_magic`"""
        # Clean any chat history
        self.conversation = []

    def _handle_hf_home_magic(self, *args):
        """Handle `%hf_home` magic command. Invoked by `_handle_magic`

        Args:
            args (list): The target HF_HOME path. Only use the first position
        """
        if not args:
            stream_content = {"name": "stdout", "text": f"HF_HOME = {self.cache_dir}"}
            self.send_response(self.iopub_socket, "stream", stream_content)
        else:
            self.cache_dir = args[0]

    def _handle_model_list_magic(self):
        """Handle `%model_list` magic command. Invoked by `_handle_magic`"""
        # Parse directories in the cache_dir
        models = os.listdir(os.path.join(self.cache_dir, "hub"))
        output = "\n - ".join(["/".join(m.split("--")[1:]) for m in models if m.startswith("models")])
        output = f"Available models:\n - {output}"
        display_content = {"data": {"text/markdown": output}, "metadata": {}}
        self.send_response(self.iopub_socket, "display_data", display_content)

    def _load_model(self):
        """Instantiate a LLM class"""
        # https://github.com/vllm-project/vllm/issues/9727
        if self.model:
            stream_content = {
                "name": "stdout",
                "text": f"A model has been loaded. Restarting kernel is needed to load another model\n",
            }
            self.send_response(self.iopub_socket, "stream", stream_content)
            return

        config = AutoConfig.from_pretrained(self.model_id)
        quantization_config = getattr(config, "quantization_config", None)
        if "bitsandbytes" in str(quantization_config):
            quantization = "bitsandbytes"
            load_format = "bitsandbytes"
        else:
            quantization = None
            load_format = "auto"

        vision_config = getattr(config, "vision_config", None)
        if vision_config:
            limit_mm_per_prompt = {"image": self.chatbot_config.n_images}
            enforce_eager = True
            max_model_len = 4096
            max_num_seqs = 16

            stream_content = {
                "name": "stdout",
                "text": f"Vision model cannot support streaming well. You have to set n_new_tokens to a large number\n",
            }
            self.send_response(self.iopub_socket, "stream", stream_content)
        else:
            limit_mm_per_prompt = None
            enforce_eager = None
            max_model_len = None
            max_num_seqs = None

        self.model = LLM(
            model=self.model_id,
            dtype=self.chatbot_config.dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            tensor_parallel_size=self.chatbot_config.tensor_parallel_size,
            limit_mm_per_prompt=limit_mm_per_prompt,
            allowed_local_media_path=os.getenv("HOME"),
            quantization=quantization,
            load_format=load_format,
        )
