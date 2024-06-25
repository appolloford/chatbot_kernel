from ipykernel.kernelapp import IPKernelApp
from . import ChatbotKernel

IPKernelApp.launch_instance(kernel_class=ChatbotKernel)
