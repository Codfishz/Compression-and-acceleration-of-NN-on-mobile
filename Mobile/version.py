import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

#device = 'cuda' if torch.cuda.is_available() else 'cpu'


#model.load_state_dict(torch.load("/home/xdd44/Desktop/Codfish/decomposition/checkpoint/test0319_module_40_CP.pth"))
#model=torch.jit.load("/Users/Rui/Documents/Python/Switch/qat.pth")
##model=model.to(device)
#model.eval()
#scripted_module = torch.jit.script(model)
## Export full jit version model (not compatible mobile interpreter), leave it here for comparison
#scripted_module.save("deeplabv3_scripted.pt")
## Export mobile interpreter version model (compatible with mobile interpreter)
#optimized_scripted_module = optimize_for_mobile(scripted_module)
#optimized_scripted_module._save_for_lite_interpreter("deeplabv3_scripted.ptl")
from torch.jit.mobile import (
    _backport_for_mobile,
    _get_model_bytecode_version,
)

MODEL_INPUT_FILE = "/Users/Rui/Documents/Python/Switch/0328.ptl"
MODEL_OUTPUT_FILE = "/Users/Rui/Documents/Python/Switch/model_v5.ptl"

print("model version", _get_model_bytecode_version(f_input=MODEL_INPUT_FILE))

_backport_for_mobile(f_input=MODEL_INPUT_FILE, f_output=MODEL_OUTPUT_FILE, to_version=5)

print("new model version", _get_model_bytecode_version(MODEL_OUTPUT_FILE))
