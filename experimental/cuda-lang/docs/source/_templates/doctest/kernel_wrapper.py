import cuda.lang as cl
import torch

@cl.kernel
def kernel():
    # begin-snippet
    {{body}}
    # end-snippet


torch.cuda.init()
cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())
torch.cuda.synchronize()

