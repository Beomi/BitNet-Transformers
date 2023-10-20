# 0️⃣1️⃣🤗 BitNet-Transformers: Huggingface Transformers Implementation of "BitNet: Scaling 1-bit Transformers for Large Language Models" in pytorch with Llama(2) Architecture

![BitNet Architecture](./static/bitnet-arch.png)

![BitNet](./static/bitnet.png)

- Paper Link: https://arxiv.org/pdf/2310.11453.pdf

## Prepare Dev env

```bash
# Clone this repo
git clone https://github.com/beomi/bitnet-transformers
cd bitnet-transformers

# Clone transformers repo
git clone https://github.com/huggingface/transformers
pip install -e transformers

# Update Llama(2) model
rm ./transformers/src/transformers/models/llama/modeling_llama.py
ln -s $(pwd)/bitnet_llama/modeling_llama.py ./transformers/src/transformers/models/llama/modeling_llama.py
```

We'll overwrite `bitnet_llama/modeling_llama.py` into `transformers`. Since the file is linked, any changes made to the file will be reflected in the `transformers` repo.


## GPU Mem Usage Comparison

**Original LLAMA**

```bash
[1] NVIDIA H100 PCIe | 58°C,   0 % | 49023 / 81559 MB | datadriven(49010M)
```

**BitLLAMA**

- Use bf16(or fp16) on-the-fly when needed

```bash
[1] NVIDIA H100 PCIe | 62°C, 100 % | 48289 / 81559 MB | datadriven(48276M)
```

