import cuda.lang as cl
import torch

torch.cuda.init()
stream = torch.cuda.current_stream()

# begin-snippet
{{body}}
# end-snippet

torch.cuda.synchronize()
