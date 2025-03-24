# 1. Download the Deepseek R1
For this I chose the 1.5 B parameter version as I don't have that much compute, but you can choose any model size you want as long as your equipment can handle it.

## A. Just make sure you find the model on hugging face and clone the files to your system.
On the model page go to the files and versions tab and click on the three dots in the upper right corner and then the "clone repository" button.

For 1.5B I did:

`git lfs install`

`https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/main`

![alt text](hf-download-deepseek.png =300x200)

## B. You will likely need download the weights separately.
So on that same page find the model safetensors download button (highlighted in the picture) and copy the link address.

![alt text](hf-download-btn.png =300x200)

Then run `wget -O model.safetensors {link}` or for 1.5B parameter model:

`wget -O model.safetensors https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/model.safetensors?download=true`