import cuda.lang as cl
import torch

@cl.kernel
def kernel(array):
    # begin-snippet
    {{body}}
    # end-snippet


torch.cuda.init()
array = torch.zeros(3, 3, dtype=torch.uint32, device="cuda")
cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (array,))
torch.cuda.synchronize()

