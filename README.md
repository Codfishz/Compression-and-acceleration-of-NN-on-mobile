# Compression and acceleration of nerual network on mobile device

In modern society, the importance of mobile electronic devices in people's lives is becoming more and more prominent. It is an excellent choice to install neural networks on mobile terminals to make mobile electronic devices more intelligent, convenient and deal with more problems.

As high-quality solutions for complex issues such as image recognition and classification, many scholars' research has focused on neural networks in recent years. There are many different methods and corresponding research results on improving the speed and reducing the size of neural networks. However, most of the related research is carried out in the PC (GPU) environment, and there is a large gap between the hardware conditions of mobile devices and PCs. The performance of existing methods is still unclear for neural networks deployed on mobile terminals without GPU support and large memory size.

This project will compress and test a typical neural network to observe the model's size, accuracy, and running time on the mobile terminal and compare the advantages and disadvantages of various standard neural network compression methods.


The "train" folder contains the code of training original model.
The "quantization" folder contains the code of quantization operation (quantization.py for static quantization, quantization2.py for quantization aware training)
The "pruning" folder contains the code of pruning operation. "pruning.py" I test the pruning method provided by pytorch official But since it could not reduce model size(would set the weight to 0 rather delete the node), later test would be done by torch-pruning package (pruning_torch.py). The retrain.py is used to retrain the model after pruning.=
The "decomposition" folder contains the code of low rank approximation. The methods I test including CP-decomposition and Tucker-decomposition. The retrain.py is used to retrain the model after pruning.
The "mobile" folder contains the source code of mobile application I used to test the running time (Test folder). The "version.py" and "switch_mobile.py" are used for optimize the model, making it able to run on mobile device.
The  "result" folder contains some screenshoots of accuacy result. Due to the size limitation, the model file could not be uploaded. The comparison of model sizes can be seen in "model size".
