import os
import unittest
import jupyter_kernel_test as jkt
from unittest import SkipTest


class ChatbotKernelTests(jkt.KernelTests):

    # REQUIRED

    kernel_name = "chatbot"

    # OPTIONAL

    # the name of the language the kernel executes
    # checked against language_info.name in kernel_info_reply
    language_name = "chatbot"

    # the normal file extension (including the leading dot) for this language
    # checked against language_info.file_extension in kernel_info_reply
    file_extension = ".txt"

    # code which should write the exact string `hello, world` to STDOUT
    code = "hi"

    def _get_local_model_list(self):
        # Get default cache_dir
        default_home = os.path.join(os.path.expanduser("~"), ".cache")
        HF_HOME = os.path.expanduser(
            os.getenv(
                "HF_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
            )
        )

        models = os.listdir(os.path.join(HF_HOME, "hub"))
        model_names = ["/".join(m.split("--")[1:]) for m in models if m.startswith("models")]
        return model_names

    def test_help(self):
        self.flush_channels()
        reply, output_msgs = self.execute_helper(code='%help')
        self.assertEqual(output_msgs[0]["header"]["msg_type"], "display_data")
        for command in ["help", "config", "load", "hf_home", "model_list", "new_chat"]:
            item = f"<tr><td style='text-align: left;'>{command}</td>"
            self.assertIn(item, output_msgs[0]["content"]["data"]["text/markdown"])

    def test_config_and_load(self):
        self.flush_channels()
        models = self._get_local_model_list()
        if not models:
            raise SkipTest("No local LLM is found")

        self.execute_helper(code='%config temperature 0.8')
        self.execute_helper(code='%config dtype float16')
        reply, output_msgs = self.execute_helper(code=f'%load {models[0]}\nhi')
        self.assertTrue(len(output_msgs[-1]["content"]["data"]["text/markdown"]) > 0)

if __name__ == "__main__":
    unittest.main()
