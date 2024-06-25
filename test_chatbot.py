import unittest

import jupyter_kernel_test as jkt


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


if __name__ == "__main__":
    unittest.main()
