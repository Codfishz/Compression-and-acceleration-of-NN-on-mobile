# Compression
Compression and acceleration of nerual network

The "train" folder contains the code of training original model.

The "quantization" folder contains the code of quantization operation (quantization.py for static quantization, quantization2.py for quantization aware training)

The "pruning" folder contains the code of pruning operation. "pruning.py" I test the pruning method provided by pytorch official But since it could not reduce model size(would set the weight to 0 rather delete the node), later test would be done by torch-pruning package (pruning_torch.py). The retrain.py is used to retrain the model after pruning.

The "decomposition" folder contains the code of low rank approximation. The methods I test including CP-decomposition and Tucker-decomposition. The retrain.py is used to retrain the model after pruning.

The "mobile" folder contains the source code of mobile application I used to test the running time (Test folder). The "version.py" and "switch_mobile.py" are used for optimize the model, making it able to run on mobile device.

The  "result" folder contains some screenshoots of accuacy result. Due to the size limitation, the model file could not be uploaded. The comparison of model sizes can be seen in "model size".
