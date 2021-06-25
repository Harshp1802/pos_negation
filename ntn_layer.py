import torch
import torch.nn as nn
import numpy as np
import random
SEED = 42
import math

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class NeuralTensorLayer(nn.Module):

  def __init__(self, output_dim, input_dim=None):
    super(NeuralTensorLayer, self).__init__()
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d   
    k = self.output_dim
    d = self.input_dim
    W = torch.empty(size=(k,d,d))
    self.W = nn.Parameter(nn.init.normal_(W))
    V = torch.empty(size=(2*d,k))
    self.V = nn.Parameter(nn.init.normal_(V))
    self.b = nn.Parameter(torch.zeros((self.input_dim,)))
    self.trainable_weights = [self.W, self.V, self.b]

  def forward(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) <= 1:
      raise Exception('BilinearTensorLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = e1.shape[0]
    k = self.output_dim
    feed_forward_product = torch.matmul(torch.cat([e1,e2], axis=-1), self.V)
    bilinear_tensor_products = []
    for i in range(k):
      btp = torch.sum((e2 * torch.matmul(e1, self.W[i])) + self.b, axis=-1)
      bilinear_tensor_products.append(btp)
    result = torch.tanh(torch.reshape(torch.cat(bilinear_tensor_products, axis=0), (batch_size, -1,k)) + feed_forward_product)
    return result