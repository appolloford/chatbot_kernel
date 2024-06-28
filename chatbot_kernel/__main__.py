import sys
from ipykernel.kernelapp import IPKernelApp
from . import ChatbotKernel
from .kernelspec import main as install_main

if len(sys.argv) > 1 and sys.argv[1] == 'install':
    install_main(sys.argv[2:])
else:
    IPKernelApp.launch_instance(kernel_class=ChatbotKernel)

