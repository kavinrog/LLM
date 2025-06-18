# AdaDecode ✨

**Faster Large Language Model (LLM) Decoding using Adaptive Layer Parallelism**

![Paper](https://arxiv.org/pdf/2506.03700)

---

## 🚀 What is AdaDecode?

**AdaDecode** is a decoding technique for LLMs that makes text generation **faster** without changing the model or using extra helper models. It works by **predicting some tokens early** (before all layers are done) and starting the next token's processing right away.

> "If the model is already confident about the next word, let's not wait. Just go!"
