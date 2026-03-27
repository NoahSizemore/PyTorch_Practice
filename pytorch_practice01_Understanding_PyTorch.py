
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    scalar = torch.tensor(7)
    scalar.ndim
    vector = torch.tensor([7, 7])
    vector.ndim
    MATRIX = torch.tensor([[7, 8],[9, 10]])
    MATRIX.ndim
    TENSOR = torch.tensor([[[1, 2, 3,],[3, 6, 9],[2, 4, 5]]])
    TENSOR.ndim

    random_tensor = torch.rand(3, 4)
    random_tensor_image = torch.rand(size=(224, 224, 3))
    ones = torch.ones(size=(3, 4))

    range = torch.arange(start=0, end=1000, step = 77)
    like = torch.zeros_like(input=range)

    float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None, # data type
                                                    device="cpu", # physical, can live in cpu or gpu
                                                    requires_grad=False) # tracks gradients with operation                             
    int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32) 

    custom_rand_tensor = torch.rand(size=(3, 4), dtype=None, device="cpu")            

    """
    - Adding, multiplying, subtraction, and division of a scalar to a tensor returns it to every
    value in the tensor (i.e., +10 for a [1, 2] tensor returns [11, 12])

    - Functions exsist for these operations for clear look: torch.mul(tensor, number)

    - torch.matmul() is matrix multiplication

    - Reshaping is to change tensor input to a defined shape
    - View shows the same tensor from a different shape
    - Stacking is combining multiple tensors on top of each other, vstack, or side-by-side, hstack.
    - Squeeze removes one dimention, unsqueeze adds a dimension
    - Permute returns a view of the input with dimensions permuted, or swapped, in a certain way
    
    """

    x = torch.arange(1. , 10.)
    x_reshape = x.reshape(9, 1)
    x_stack = torch.stack([x, x, x, x], dim=1)

    x_rereshape = x.reshape(1, 3, 3)

    array = np.arange(1.0, 8.0)
    tenarray=torch.from_numpy(array)

    custom_rand_tensor2 = torch.rand(size=(3, 4), dtype=None, device="cpu")  
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    custom_rand_tensor3 = torch.rand(size=(3, 4), dtype=None, device="cpu")  
    torch.manual_seed(RANDOM_SEED)
    custom_rand_tensor4 = torch.rand(size=(3, 4), dtype=None, device="cpu")  

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Note: numpy on works with the cpu, use .cpu() to change device
    test_tensor_gpu = torch.tensor([1, 2, 3], dtype=None, device=device)


    """ print(scalar)
    print(vector.shape)
    print(MATRIX.shape)
    print(TENSOR.shape)
    print(TENSOR.ndim)
    print(random_tensor)
    print(random_tensor_image.shape,  random_tensor_image.ndim)
    print(ones.dtype)
    print(range)
    print(like)
    print(float_32_tensor.dtype)
    print(float_32_tensor * int_32_tensor)
    print(int_32_tensor.device)
    print(custom_rand_tensor * custom_rand_tensor)
    print(torch.matmul(custom_rand_tensor, custom_rand_tensor))

    print(custom_rand_tensor.min())
    print(custom_rand_tensor.max())
    print(custom_rand_tensor.mean())
    print(custom_rand_tensor.sum())
    print(custom_rand_tensor)
    print(custom_rand_tensor.argmin())
    print(custom_rand_tensor.argmax())

    print(x, x.shape, "\n", x_reshape, x_reshape.shape)
    print(x.view(1, 9)) # changing this would change what x is since it shares the same memory
    print(x.shape)
    print(x.unsqueeze(dim=0).shape)
    print(random_tensor_image.permute(dims=(2, 0, 1)).shape)
    print(x_rereshape, x_rereshape[0], x_rereshape[0, 0], x_rereshape[0, 0, 0])

    print(array, tenarray, tenarray.numpy())

    print(custom_rand_tensor, "\n", custom_rand_tensor2, "\n", custom_rand_tensor == custom_rand_tensor2)
    print(custom_rand_tensor3, "\n", custom_rand_tensor4, "\n", custom_rand_tensor3 == custom_rand_tensor4)"""
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(test_tensor_gpu, test_tensor_gpu.device, test_tensor_gpu.cpu().numpy())
    


    exit(0)

if __name__ == "__main__":
    main()