#===============================================================================
# /* src/my_lib.h */
# int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output);
# int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
#  
#  
# /* src/my_lib.c */
# #include <TH/TH.h>
#  
# int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
# THFloatTensor *output)
# {
#     if (!THFloatTensor_isSameSizeAs(input1, input2))
#         return 0;
#     THFloatTensor_resizeAs(output, input1);
#     THFloatTensor_add(output, input1, input2);
#     return 1;
# }
#  
# int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
# {
#     THFloatTensor_resizeAs(grad_input, grad_output);
#     THFloatTensor_fill(grad_input, 1);
#     return 1;
# }
#  
#  
# # build.py
# from torch.utils.ffi import create_extension
# ffi = create_extension(
# name='_ext.my_lib',
# headers='src/my_lib.h',
# sources=['src/my_lib.c'],
# with_cuda=False
# )
# ffi.build()
#  
#  
# # functions/add.py
# import torch
# from torch.autograd import Function
# from _ext import my_lib
#  
#  
# class MyAddFunction(Function):
#     def forward(self, input1, input2):
#         output = torch.FloatTensor()
#         my_lib.my_lib_add_forward(input1, input2, output)
#         return output
#  
#     def backward(self, grad_output):
#         grad_input = torch.FloatTensor()
#         my_lib.my_lib_add_backward(grad_output, grad_input)
#         return grad_input
#      
#  
# # modules/add.py
# from torch.nn import Module
# from fadd import MyAddFunction
#  
# class MyAddModule(Module):
#     def forward(self, input1, input2):
#         return MyAddFunction()(input1, input2)
#      
#      
# # main.py
# import torch.nn as nn
# from torch.autograd import Variable
# from madd import MyAddModule
#  
# class MyNetwork(nn.Module):
#     def __init__(self):
#         super(MyNetwork, self).__init__(
#             add=MyAddModule(),
#         )
#  
#     def forward(self, input1, input2):
#         return self.add(input1, input2)
#  
# model = MyNetwork()
# input1, input2 = Variable(torch.randn(5, 5)), Variable(torch.randn(5, 5))
# print(model(input1, input2))
# print(input1 + input2)
#===============================================================================