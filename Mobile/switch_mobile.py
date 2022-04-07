import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#model.load_state_dict(torch.load("/home/xdd44/Desktop/Codfish/decomposition/checkpoint/test0319_module_40_CP.pth"))
model=torch.jit.load("/home/xdd44/Desktop/Codfish/Pruning and quantization/prun20_sta.pth")
#model=model.to(device)
model.eval()
torchscript_model = torch.jit.script(model)
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, "/home/xdd44/Desktop/Codfish/mobile/model_mobile_prun20_sat.pt")