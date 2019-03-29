# Face-Reenactment
Allows to train DNN to reenact face from source to destination
This is based on widely available framework of recognizing and learning face using multi-layer neural network. This particular approach 
introduces new model and allows for dynamic vram growth.  Was tested on Nvidia Titan V(12GB) and Intel I9-7980XE. You can adapt the scripts to
run in linux. Tested on Ubuntu. Speed ov neural network training is faster on Ubuntu vs Windows 10 (27% faster). Same machine. More headache to 
configure. You will need tensorflow and keras, nvidia cudnn, python installed in anaconda. 
