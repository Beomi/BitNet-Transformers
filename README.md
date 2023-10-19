# BitNet-Llama: Implementation of "BitNet: Scaling 1-bit Transformers for Large Language Models" in pytorch with Llama(2) Architecture

- Paper Link: https://arxiv.org/pdf/2310.11453.pdf

## Prepare Dev env

```bash
git clone https://github.com/huggingface/transformers
pip install -e transformers
ln -s transformers/src/transformers/models/llama/modeling_llama.py .
```

We're using `modeling_llama.py` from `transformers`. Since the file is linked, any changes made to the file will be reflected in the `transformers` repo.

