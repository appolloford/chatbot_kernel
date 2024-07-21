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

    def test_config_dtype(self):
        self.flush_channels()
        reply, output = self.execute_helper(code="%config dtype float16")
        self.assertEqual(reply['content']['status'], 'ok')
        self.assertEqual(output, [])
        reply, output = self.execute_helper(code="%config dtype 16")
        self.assertEqual(reply['content']['status'], 'error')
        self.assertEqual(output[0]['content']['ename'], repr(ValueError))
        self.assertIn('Invalid dtype', output[0]['content']['evalue'])

    def test_config_n_predict(self):
        self.flush_channels()
        reply, output = self.execute_helper(code="%config n_predict 100")
        self.assertEqual(reply['content']['status'], 'ok')
        self.assertEqual(output, [])
        reply, output = self.execute_helper(code="%config n_predict abc")
        self.assertEqual(reply['content']['status'], 'error')
        self.assertEqual(output[0]['content']['ename'], repr(ValueError))

    def test_config_n_new_tokens(self):
        self.flush_channels()
        reply, output = self.execute_helper(code="%config n_new_tokens 100")
        self.assertEqual(reply['content']['status'], 'ok')
        self.assertEqual(output, [])
        reply, output = self.execute_helper(code="%config n_new_tokens abc")
        self.assertEqual(reply['content']['status'], 'error')
        self.assertEqual(output[0]['content']['ename'], repr(ValueError))

    def test_config_temperature(self):
        self.flush_channels()
        reply, output = self.execute_helper(code="%config temperature 0.1")
        self.assertEqual(reply['content']['status'], 'ok')
        self.assertEqual(output, [])
        reply, output = self.execute_helper(code="%config temperature 10")
        self.assertEqual(reply['content']['status'], 'error')
        self.assertEqual(output[0]['content']['ename'], repr(ValueError))
        reply, output = self.execute_helper(code="%config temperature abc")
        self.assertEqual(reply['content']['status'], 'error')
        self.assertEqual(output[0]['content']['ename'], repr(ValueError))

    def test_config_undefined(self):
        self.flush_channels()
        reply, output = self.execute_helper(code="%config undefined xxx")
        self.assertEqual(reply['content']['status'], 'error')
        self.assertEqual(output[0]['content']['ename'], repr(ValueError))

    def test_help(self):
        self.flush_channels()
        reply, output = self.execute_helper(code='%help')
        self.assertEqual(output[0]["header"]["msg_type"], "display_data")
        for command in ["help", "config", "load", "hf_home", "model_list", "new_chat"]:
            item = f"<tr><td style='text-align: left;'>{command}</td>"
            self.assertIn(item, output[0]["content"]["data"]["text/markdown"])

    def test_load(self):
        self.flush_channels()
        models = self._get_local_model_list()
        if not models:
            raise SkipTest("No local LLM is found")

        reply, output = self.execute_helper(code=f'%load {models[0]}\nhi')
        print(output)
        self.assertEqual(reply['content']['status'], 'ok')
        self.assertTrue(len(output[-1]["content"]["data"]["text/markdown"]) > 0)

if __name__ == "__main__":
    unittest.main()
