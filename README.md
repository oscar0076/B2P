# B2P: From Buildings to People
A Large Vision-Language Model for Urban Density Estimation

**Note:** This project is a fine-tune of the [GeoChat](https://github.com/mbzuai-oryx/GeoChat) model, adapted for urban population density estimation.
## Install
```bash
# Clone the repository
git clone https://github.com/oscar0076/B2P.git
# Change directory
B2P
#
conda create -n b2p python=3.10 -y
conda activate b2p
pip install -r requires.txt
#For training cases
pip install ninja
```
## Project architecture
The project consists of a visual encoder, a projector layer, and a large language model (LLM). The visual encoder uses a pretrained CLIP-14-336px, the projector layer is composed of MLP layers used to adapt vision-tokens to language space suitable for input to a Large Language Model(Vicuna 1.5). 


The model is trained in two stages (Pre-training and Fine-tuning)

